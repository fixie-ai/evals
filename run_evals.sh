#!/bin/bash
EVALS_THREADS=2 oaievalset generation/direct/ultravox-70b audio-transcription-set --log_to_file results/ultravox-70b/audio-transcription.txt --record_dir results/ultravox-70b/ --max_samples 5000
EVALS_THREADS=2 oaievalset generation/direct/ultravox-8b audio-transcription-set --log_to_file results/ultravox-8b/audio-transcription.txt --record_dir results/ultravox-8b/ --max_samples 5000
# EVALS_THREADS=10 oaievalset generation/whisper audio-transcription-set --log_to_file results/whisper/audio-transcription.txt --record_dir results/whisper/ --max_samples 5000
# EVALS_THREADS=10 oaievalset generation/deepgram/nova-3 audio-transcription-set --log_to_file results/deepgram-nova-3/audio-transcription.txt --record_dir results/deepgram-nova-3/ --max_samples 5000
EVALS_THREADS=10 oaievalset generation/scribe audio-transcription-set --log_to_file results/scribe/audio-transcription.txt --record_dir results/scribe/ --max_samples 5000

EVALS_THREADS=10 oaievalset generation/whisper ami-ihm-transcription-set --log_to_file results/whisper/ami-ihm-transcription.txt --record_dir results/whisper/ --max_samples 5000
EVALS_THREADS=10 oaievalset generation/deepgram/nova-3 ami-ihm-transcription-set --log_to_file results/deepgram-nova-3/ami-ihm-transcription.txt --record_dir results/deepgram-nova-3/ --max_samples 5000

echo "Done"