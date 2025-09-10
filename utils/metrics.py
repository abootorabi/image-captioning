#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation metrics for image captioning.
This module implements metrics for evaluating caption quality.
"""

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
#from vocabulary import Vocabulary

# Make sure NLTK tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def calculate_bleu(references, hypotheses, vocab , max_n=4):
    """
    Calculate BLEU score for a set of references and hypotheses.
    
    Args:
        references (list): List of reference lists (multiple references per sample)
        hypotheses (list): List of hypothesis lists (one per sample)
        max_n (int): Maximum n-gram to consider
        
    Returns:
        list: BLEU scores for different n-grams (BLEU-1, BLEU-2, etc.)
    """
    
    bleu_scores = []
    smoothing= SmoothingFunction().method1
    for i in range(len(references)):
        for j in range(len(references[i])):
            reference = references[i][j]
            
            if type(reference) != list :
                references[i][j] = vocab.tokenize(reference)
            
        if type(hypotheses[i]) != list :
            hypotheses[i] = vocab.tokenize(hypotheses[i])
            
    # Calculate BLEU-1 to BLEU-max_n
    for n in range(1, max_n+1):
        weights = tuple((1.0 / n if i < n else 0.0) for i in range(max_n))
        bleu = corpus_bleu(references, hypotheses, weights=weights, smoothing_function=smoothing)
        bleu_scores.append(bleu)
    print(references[:3] , hypotheses[:3])
    print('bleu_scores' , bleu_scores)
    return bleu_scores



    
def calculate_metrics(model, dataloader, vocab, device='cuda', max_samples=None, beam_size=1):
    """
    Calculate evaluation metrics for the captioning model.

    Args:
        model (nn.Module): Image captioning model
        dataloader (DataLoader): Data loader (should return image, caption, image_id)
        vocab (Vocabulary): Vocabulary object
        device (str): Device to use ('cuda' or 'cpu')
        max_samples (int, optional): Maximum number of samples to evaluate
        beam_size (int): Beam size for caption generation

    Returns:
        float: BLEU-4 score
    """
    model.eval()

    references= []
    hypotheses = []

    sample_count = 0

    with torch.no_grad():
        for images , captions in tqdm(dataloader, desc="Generating captions"):
            batch_size = images.size(0)

            if max_samples is not None and sample_count >= max_samples:
                break

            images = images.to(device)

            for j in range(batch_size):
                if max_samples is not None and sample_count >= max_samples:
                    break

                image = images[j].unsqueeze(0)  # [1, 3, H, W]
                caption = captions[j]

                # Generate caption (assuming model can only process one image at a time)
                predicted_ids = model.generate_caption(
                    image,
                    beam_size=beam_size
                )

                # Decode generated caption
                predicted_caption = vocab.decode(predicted_ids[0].tolist(), join=True, remove_special=True)

                # Decode reference caption
                reference_caption = vocab.decode(caption.tolist(), join=True, remove_special=True)

                # Store captions 
                references.append([reference_caption])
                hypotheses.append(predicted_caption)

                sample_count += 1


    print('hypotheses:', hypotheses[-1])

    # Calculate BLEU scores
    
    bleu_scores = calculate_bleu(references, hypotheses , vocab)

    return bleu_scores[-1]
