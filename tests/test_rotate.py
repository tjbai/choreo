import torch
from unittest import TestCase
import fairscale.nn.model_parallel.initialize as fs_init
from llama.model import precompute_freqs_cis, apply_rotary_emb, reposition_rotary_emb, Transformer, ModelArgs

class TestRotate(TestCase):

    def setUp(self):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="gloo",
                init_method="tcp://localhost:12345",
                world_size=1,
                rank=0
            )
        if not fs_init.model_parallel_is_initialized():
            fs_init.initialize_model_parallel(1)

    def setup_reposition(self):
        torch.manual_seed(42)
        n_layers, batch_size, seq_len, n_heads, head_dim = 16, 10, 4, 4, 8
        xk = torch.randn(n_layers, batch_size, seq_len, n_heads, head_dim)
        freqs_cis = precompute_freqs_cis(head_dim, 48)
        return xk, freqs_cis

    def test_reposition_identity(self):
        xk, freqs_cis = self.setup_reposition()
        xk_moved = reposition_rotary_emb(xk, torch.ones(4).long(), torch.ones(4).long(), freqs_cis)
        self.assertEqual(xk_moved.shape, xk.shape)
        self.assertTrue(torch.allclose(xk_moved, xk))

    def test_reposition_forwards_backwards(self):
        xk, freqs_cis = self.setup_reposition()
        xk1 = reposition_rotary_emb(xk, torch.zeros(4).long(), torch.ones(4).long(), freqs_cis)
        xk2 = reposition_rotary_emb(xk1, torch.ones(4).long(), torch.zeros(4).long(), freqs_cis)
        self.assertTrue(torch.allclose(xk2, xk))

    def test_reposition_cycle(self):
        xk, freqs_cis = self.setup_reposition()
        pos0 = torch.zeros(4).long()
        pos1 = torch.ones(4).long()
        pos2 = torch.randint(10, (4,))

        xk1 = reposition_rotary_emb(xk, pos0, pos1, freqs_cis)
        xk2 = reposition_rotary_emb(xk1, pos1, pos2, freqs_cis)
        xk1_back = reposition_rotary_emb(xk2, pos2, pos1, freqs_cis)
        xk0_back = reposition_rotary_emb(xk1_back, pos1, pos0, freqs_cis)
        self.assertTrue(torch.allclose(xk0_back, xk, atol=1e-5))

        from_pos = torch.tensor([0, 1, 2, 3])
        to_pos = torch.tensor([3, 1, 0, 2])
        xk_moved = reposition_rotary_emb(xk, from_pos, to_pos, freqs_cis)
        xk_back = reposition_rotary_emb(xk_moved, to_pos, from_pos, freqs_cis)
        self.assertTrue(torch.allclose(xk_back, xk, atol=1e-5))

    def test_reposition_composition(self):
        xk, freqs_cis = self.setup_reposition()
        start_pos = torch.tensor([0, 1, 2, 3])
        mid_pos = torch.tensor([1, 2, 3, 4])
        final_pos = torch.tensor([2, 3, 4, 5])
        xk2 = reposition_rotary_emb(xk, start_pos, mid_pos, freqs_cis)
        xk2 = reposition_rotary_emb(xk2, mid_pos, final_pos, freqs_cis)
        xk_direct = reposition_rotary_emb(xk, start_pos, final_pos, freqs_cis)
        self.assertTrue(torch.allclose(xk2, xk_direct, atol=1e-5))

    def test_reposition_equality(self):
        xk, freqs_cis = self.setup_reposition()
        start_pos = torch.zeros(4).long()
        mid_pos = torch.tensor([1, 2, 3, 4])
        final_pos = torch.tensor([2, 3, 4, 5])

        xk1 = reposition_rotary_emb(xk, start_pos, mid_pos, freqs_cis)
        _, xk1_apply = apply_rotary_emb(torch.zeros_like(xk), xk, freqs_cis[mid_pos])
        self.assertTrue(torch.allclose(xk1, xk1_apply, atol=1e-5))

        xk2 = reposition_rotary_emb(xk1, mid_pos, final_pos, freqs_cis)
        _, xk2_apply = apply_rotary_emb(torch.zeros_like(xk), xk, freqs_cis[final_pos])
        self.assertTrue(torch.allclose(xk2, xk2_apply, atol=1e-5))

    def setup_transformer(self):
        params = ModelArgs(
            dim=32,
            n_layers=2,
            n_heads=4,
            vocab_size=100,
            max_batch_size=1,
            max_seq_len=8,
            rope_theta=10000
        )
        return Transformer(params)

    def test_reposition_cache(self):
        model = self.setup_transformer()

        torch.manual_seed(42)
        cache_k_original = torch.randn_like(model.cache_k)
        model.cache_k.copy_(cache_k_original)

        where = torch.tensor([1, 3, 5])
        from_pos = torch.tensor([1, 3, 5])
        to_pos = torch.tensor([4, 6, 2])
        model.reposition_cache(where, from_pos, to_pos)

        unmodified_pos = torch.ones(model.params.max_seq_len, dtype=bool)
        unmodified_pos[where] = False
        self.assertTrue(torch.allclose(
            model.cache_k[:, :, unmodified_pos],
            cache_k_original[:, :, unmodified_pos]
        ))

        model.reposition_cache(where, to_pos, from_pos)
        self.assertTrue(torch.allclose(model.cache_k, cache_k_original, atol=1e-5))

    def test_reposition_cache_piecewise(self):
        model = self.setup_transformer()
        torch.manual_seed(42)

        cache_k_original = torch.randn_like(model.cache_k)
        model.cache_k.copy_(cache_k_original)

        where1 = torch.tensor([1, 3])
        where2 = torch.tensor([5, 7])
        from_pos = torch.tensor([1, 3, 5, 7])
        to_pos = torch.tensor([4, 6, 2, 0])

        model.reposition_cache(where1, from_pos[0:2], to_pos[0:2])
        model.reposition_cache(where2, from_pos[2:4], to_pos[2:4])
        cache_k_piecewise = model.cache_k.clone()

        model.cache_k.copy_(cache_k_original)
        model.reposition_cache(torch.cat([where1, where2]), from_pos, to_pos)

        self.assertTrue(torch.allclose(cache_k_piecewise, model.cache_k, atol=1e-5))

        model.cache_k.copy_(cache_k_original)
        where1 = torch.tensor([1, 3, 5])
        where2 = torch.tensor([3, 5, 7])
        from_pos = torch.tensor([1, 3, 5, 6, 2, 7])
        to_pos = torch.tensor([4, 6, 2, 0, 1, 3])

        model.reposition_cache(where1, from_pos[0:3], to_pos[0:3])
        model.reposition_cache(where2, from_pos[3:6], to_pos[3:6])
        cache_k_piecewise = model.cache_k.clone()

        model.cache_k.copy_(cache_k_original)
        where_all = torch.tensor([1, 3, 5, 7])
        from_pos_final = torch.tensor([1, 3, 5, 7])
        to_pos_final = torch.tensor([4, 0, 1, 3])
        model.reposition_cache(where_all, from_pos_final, to_pos_final)
        self.assertTrue(torch.allclose(cache_k_piecewise, model.cache_k, atol=1e-5))
