# ARES v2.0 - AI Voice Detection System

A multi-stream system for detecting AI-generated or cloned voices.

>  **Note**: This project is a work in progress and not finished yet.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ARES Engine                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Stream A   │  │  Stream B   │  │  Stream C   │              │
│  │  Wav2Vec2   │  │  Bi-LSTM    │  │   SLIM      │              │
│  │ (Spectral)  │  │ (Acoustic)  │  │ (Prosody)   │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                      │
│         └────────────────┼────────────────┘                      │
│                          ▼                                       │
│               ┌──────────────────┐                               │
│               │  Fusion Model    │                               │
│               └────────┬─────────┘                               │
│                        ▼                                         │
│            AUTHENTIC / CLONED + Confidence                       │
└─────────────────────────────────────────────────────────────────┘
```

## 🔬 How It Works

**Stream A (Spectral)**: Uses pre-trained Wav2Vec2 to detect spectral artifacts from AI synthesizers.

**Stream B (Acoustic)**: Analyzes jitter and shimmer - AI voices are often too "perfect" with unnatural stability.

**Stream C (Prosody)**: Uses Whisper to detect mismatches between speaking style and linguistic content.

**Fusion**: Combines all stream scores with attention weighting to produce final classification.

