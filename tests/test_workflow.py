import torch
from unittest import TestCase
from llama.workflow import Workflow, grouped_causal_mask, incremental_sequence_with_offset

class TestWorkflow(TestCase):
    def setUp(self):
        self.workflow = Workflow.__new__(Workflow)

        class MockModel:
            def forward(self, *args, **kwargs):
                pass

            def reposition_cache(self, *args, **kwargs):
                pass

        class MockFormatter:
            def encode_message(self, message):
                return [128006, 1, 128009]

            def encode_dialog(self, dialog):
                tokens = []
                for message in dialog:
                    tokens.extend(self.encode_message(message))
                return tokens

        class MockTokenizer:
            bos_id = 128000
            stop_tokens = [128001, 128009]

        self.workflow.model = MockModel()
        self.workflow.formatter = MockFormatter()
        self.workflow.tokenizer = MockTokenizer()
        self.workflow.max_seq_len = 128
        self.workflow.device = "cpu"
        self.workflow.max_nodes = 10
        self.workflow.reset()

    def test_parent_mask(self):
        self.workflow.cache_len = 4
        self.workflow.context = torch.tensor([1, 2, 3, 4])
        self.workflow.node_map = torch.tensor([0, 1, 1, 2])
        self.workflow.cur_id = 3

        self.workflow.add_nodes([
            {'parent_ids': [1], 'expects': ('assistant', None)},
            {'parent_ids': [2], 'expects': ('assistant', None)},
            {'parent_ids': [1, 2], 'expects': ('assistant', None)}
        ]) # type: ignore
        self.assertTrue(torch.all(
            self.workflow.adj[3:6] ==
            torch.tensor([
                [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
            ], dtype=torch.bool)
        ))

        mask = self.workflow.dynamic_mask(3, 6)
        self.assertEqual(mask.shape, (3, 4))
        self.assertTrue(torch.all(
            mask == torch.tensor([
                [0, 0, 0, float("-inf")],
                [0, float("-inf"), float("-inf"), 0],
                [0, 0, 0, 0]
            ])
        ))

    def test_preallocate_interleaved_mask(self):
        bsz = 2
        max_gen_len = 3
        base_mask = torch.tensor([[0.0, float("-inf")], [float("-inf"), 0.0]])
        mask = self.workflow.preallocate_interleaved_mask(base_mask, bsz, max_gen_len)
        self.assertEqual(mask.shape, (bsz, base_mask.shape[1] + bsz * max_gen_len))
        self.assertTrue(torch.all(mask[:, :base_mask.shape[1]] == base_mask))

        interleaved = mask[:, base_mask.shape[1]:]
        for i in range(max_gen_len):
            block = interleaved[:, i*bsz:(i+1)*bsz]
            expected = torch.full((bsz, bsz), float("-inf"))
            expected.fill_diagonal_(0)
            self.assertTrue(torch.all(block == expected))

    def test_compact(self):
        self.workflow.cache_len = 6
        self.workflow.context = torch.tensor([1, 2, 3, 4, 5, 6])
        self.workflow.node_map = torch.tensor([0, 1, 1, 2, 2, 2, 3, 3, 3])
        self.workflow.position_map = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.workflow.compact(order=[2, 1], mask=torch.tensor([0, 0, 0, 0, 0, 0, float("-inf"), float("-inf"), float("-inf")]))
        self.assertTrue(torch.all(self.workflow.position_map == torch.tensor([0, 4, 5, 1, 2, 3, 6, 7, 8])))
        self.workflow.compact(order=[3, 2, 1], mask=torch.zeros(9))
        self.assertTrue(torch.all(self.workflow.position_map == torch.tensor([0, 7, 8, 4, 5, 6, 1, 2, 3])))
        self.workflow.compact(order=[3, 1, 2], mask=torch.zeros(9))
        self.assertTrue(torch.all(self.workflow.position_map == torch.tensor([0, 4, 5, 6, 7, 8, 1, 2, 3])))

    def test_compact_identity(self):
        self.workflow.cache_len = 6
        self.workflow.context = torch.tensor([1, 2, 3, 4, 5, 6])
        self.workflow.node_map = torch.tensor([0, 1, 1, 2, 2, 2, 3, 3, 3])
        self.workflow.position_map = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.workflow.compact(order=[1, 2], mask=torch.tensor([0, 0, 0, 0, 0, 0, float("-inf"), float("-inf"), float("-inf")]))
        self.assertTrue(torch.all(self.workflow.position_map == torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])))
        self.workflow.compact(order=[1, 2, 3], mask=torch.zeros(9))
        self.assertTrue(torch.all(self.workflow.position_map == torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])))

    def test_wrap_outputs(self):
        headers = [[10, 11], [12, 13]]
        tokens = torch.tensor([1, 2, 3, 128009, 5, 6])
        tasks = [
            {'parent_ids': [1], 'expects': ('assistant', None)},
            {'parent_ids': [2], 'expects': ('user', 'math')}
        ]
        out_tokens, out_nodes = self.workflow.wrap_outputs(tokens.view(-1, 2).t(), tasks, headers) # type: ignore
        self.assertEqual(out_tokens[0], [1, 3, 5])
        self.assertEqual(out_tokens[1], [2])
        self.assertEqual(len(out_nodes), len(tasks))
        self.assertEqual(out_nodes[0]['tokens'], [10, 11, 1, 3, 5])
        self.assertEqual(out_nodes[1]['tokens'], [12, 13, 2, 128009])
        self.assertEqual(out_nodes[0]['parent_ids'], tasks[0]['parent_ids'])
        self.assertEqual(out_nodes[1]['parent_ids'], tasks[1]['parent_ids'])

    def test_grouped_causal_mask(self):
        message_ids = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 2])
        mask = grouped_causal_mask(message_ids)
        self.assertEqual(mask.shape, (len(message_ids), len(message_ids)))
        self.assertTrue(torch.all(mask[:4, :4] == torch.triu(torch.full((4, 4), float("-inf")), diagonal=1)))
        self.assertTrue(torch.all(mask[4:6, 4:6] == torch.triu(torch.full((2, 2), float("-inf")), diagonal=1)))
        self.assertTrue(torch.all(mask[6:, 6:] == torch.triu(torch.full((3, 3), float("-inf")), diagonal=1)))

    def test_increment_sequence_with_offset(self):
        self.assertTrue(torch.all(
            incremental_sequence_with_offset(
                torch.tensor([10, 3, 7, 21]),
                torch.tensor([3, 1, 4, 2])
            ) == torch.tensor([10, 11, 12, 3, 7, 8, 9, 10, 21, 22])
        ))

        self.assertTrue(torch.all(
            incremental_sequence_with_offset(
                torch.tensor([5, 0, 7]),
                torch.tensor([2, 0, 3])
            ) == torch.tensor([5, 6, 7, 8, 9])
        ))

    def test_insert(self):
        [system] = self.workflow.insert([
            {
                'messages': [{'role': 'system', 'content': ''}],
                'parent_ids': []
            }
        ])

        self.assertEqual(system['id'], 1)
        self.assertEqual(self.workflow.cur_id, 2)
        self.assertTrue(torch.all(self.workflow.node_map[:4] == torch.tensor([0, 1, 1, 1])))
        self.assertTrue(torch.all(self.workflow.position_map[:4] == torch.tensor([0, 1, 2, 3])))
        self.assertTrue(torch.all(self.workflow.context[:4] == torch.tensor([128000, 128006, 1, 128009])))
        self.assertTrue(torch.all(self.workflow.adj[1, :5] == torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool)))

        user_1, user_2 = self.workflow.insert([
            {
                'messages': [{'role': 'user', 'content': ''}],
                'parent_ids': [system['id']]
            },
            {
                'messages': [{'role': 'user', 'content': ''}],
                'parent_ids': [system['id']]
            }
        ])

        self.assertEqual([user_1['id'], user_2['id']], [2, 3])
        self.assertEqual(self.workflow.cur_id, 4)
        self.assertTrue(torch.all(self.workflow.node_map[:10] == torch.tensor([0, 1, 1, 1, 2, 2, 2, 3, 3, 3])))
        self.assertTrue(torch.all(self.workflow.position_map[:10] == torch.tensor([0, 1, 2, 3, 4, 5, 6, 4, 5, 6])))
        self.assertTrue(torch.all(
            self.workflow.adj[2:4, :5] ==
            torch.tensor([
                [1, 1, 1, 0, 0],
                [1, 1, 0, 1, 0]
            ], dtype=torch.bool)
        ))

        _ = self.workflow.insert([
            {
                'messages': [{'role': 'user', 'content': ''}],
                'parent_ids': [system['id'], user_1['id']]
            },
            {
                'messages': [{'role': 'user', 'content': ''}],
                'parent_ids': [system['id'], user_1['id'], user_2['id']]
            }
        ])

        self.assertTrue(torch.all(self.workflow.position_map[10:16] == torch.tensor([7, 8, 9, 7, 8, 9])))
        self.assertTrue(torch.all(
            self.workflow.adj[4:6, :6] ==
            torch.tensor([
                [1, 1, 1, 0, 1, 0],
                [1, 1, 1, 1, 0, 1]
            ], dtype=torch.bool)
        ))

    def test_insert_multiple(self):
        [prompt] = self.workflow.insert([
            {
                'messages': [
                    {'role': 'system', 'content': ''},
                    {'role': 'user', 'content': ''},
                ],
                'parent_ids': []
            }
        ])

        self.assertEqual(prompt['id'], 1)
        self.assertEqual(self.workflow.cur_id, 2)
        self.assertTrue(torch.all(self.workflow.node_map[:7] == torch.tensor([0, 1, 1, 1, 1, 1, 1])))

    def test_leftmost_position_ids(self):
        self.workflow.cache_len = 7
        self.workflow.position_map = torch.tensor([0, 1, 2, 3, 5, 6, 7])

        mask = torch.tensor([
            [0., 0., float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf")],
            [0., 0., 0., float("-inf"), float("-inf"), float("-inf"), float("-inf")],
        ])

        pos_ids = self.workflow.leftmost_position_ids(mask == 0)
        self.assertTrue(torch.equal(pos_ids, torch.tensor([2, 3])))

        mask = torch.tensor([
            [0., float("-inf"), float("-inf"), 0., float("-inf"), float("-inf"), float("-inf")],
            [0., float("-inf"), 0., float("-inf"), 0., float("-inf"), float("-inf")],
        ])

        pos_ids = self.workflow.leftmost_position_ids(mask == 0)
        self.assertTrue(torch.equal(pos_ids, torch.tensor([4, 6])))

        mask = torch.full((2, 7), float("-inf"))
        pos_ids = self.workflow.leftmost_position_ids(mask == 0)
        self.assertTrue(torch.equal(pos_ids, torch.tensor([1, 1])))
