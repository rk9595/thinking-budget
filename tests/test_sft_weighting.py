from types import SimpleNamespace

import pytest
import torch

from train_sft import sequence_loss_callback, sequence_mean_loss


def test_equal_completion_weight_not_token_weight():
    logits = torch.zeros(2, 6, 2, requires_grad=True)
    with torch.no_grad():
        logits[0, :, 0] = 2
    labels = torch.tensor([[-100, 0, -100, -100, -100, -100],
                           [-100, 0, 0, 0, 0, 0]])
    loss = sequence_mean_loss(logits, labels)
    expected = (torch.nn.functional.softplus(torch.tensor(-2.0)) + torch.log(torch.tensor(2.0))) / 2
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None


def test_accumulated_gradients_match_equal_example_batch():
    x = torch.randn(2, 5, 3, requires_grad=True)
    labels = torch.tensor([[-100, 1, -100, -100, -100], [-100, 1, 2, 1, 2]])
    expected = torch.autograd.grad(sequence_mean_loss(x, labels), x)[0]
    holder = {"trainer": SimpleNamespace(model=SimpleNamespace(training=True),
                                        current_gradient_accumulation_steps=2)}
    callback = sequence_loss_callback(holder)
    loss = sum(callback(SimpleNamespace(logits=x[i:i+1]), labels[i:i+1], num_items_in_batch=5)
               for i in range(2))
    actual = torch.autograd.grad(loss, x)[0]
    assert torch.allclose(actual, expected)
    holder["trainer"].current_gradient_accumulation_steps = 1
    assert torch.allclose(callback(SimpleNamespace(logits=x[:1]), labels[:1]),
                          sequence_mean_loss(x[:1], labels[:1]))


def test_no_completion_labels_is_an_error():
    with pytest.raises(ValueError, match="completion token"):
        sequence_mean_loss(torch.zeros(1, 3, 2), torch.full((1, 3), -100))


def test_real_sft_trainer_handles_final_partial_accumulation(tmp_path):
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, set_seed
    from trl import SFTConfig, SFTTrainer
    raw = Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1, "<pad>": 2, "q": 3, "a": 4},
                              unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>",
                                  eos_token="<eos>", pad_token="<pad>")
    set_seed(42)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=5, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        eos_token_id=1, pad_token_id=2))
    ds = Dataset.from_list([{"prompt": "q ", "completion": "a " * n + "<eos>"} for n in [1, 2, 3]])
    holder, accumulation = {}, []
    callback = sequence_loss_callback(holder)

    def loss(*args, **kwargs):
        accumulation.append(holder["trainer"].current_gradient_accumulation_steps)
        return callback(*args, **kwargs)

    trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=ds, compute_loss_func=loss,
        args=SFTConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, fp16=False,
                       num_train_epochs=1, per_device_train_batch_size=1,
                       gradient_accumulation_steps=2, completion_only_loss=True,
                       loss_type="nll",
                       save_strategy="no", report_to="none", max_length=16))
    holder["trainer"] = trainer
    trainer.train()
    assert trainer.state.global_step == 2
    assert accumulation == [2, 2, 1]
