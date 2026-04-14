from transformers import AutoTokenizer, BertModel, RobertaModel
import os

def get_tokenlizer(text_encoder_type):
    # text_encoder_type should be:
    # - "bert-base-uncased"
    # - or your custom folder: "/path/to/custom_bert"
    print("final text_encoder_type:", text_encoder_type)

    # MUST USE FAST TOKENIZER (for char_to_token)
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_type, use_fast=True)

    return tokenizer


def get_pretrained_language_model(text_encoder_type):
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_type, use_fast=True)

    # Load custom or original BERT
    model = BertModel.from_pretrained(text_encoder_type)

    print(f"[BERT] Loaded vocab size: {model.config.vocab_size}")
    # ❗ DO NOT RESIZE — vocab size must match embeddings

    return model
