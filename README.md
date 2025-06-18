# ARES (Acoustic Resonance & Entity-Specific) Voice Verification System

ARES is a multi-layered, hybrid system for detecting AI-generated or cloned voices with high accuracy. It operates on the principle that while a cloned voice can mimic primary vocal characteristics, it fails to perfectly replicate subtle acoustic, biological, and linguistic features inherent in human speech.

## Features

- **Multi-Stream Analysis**: Combines three parallel analysis streams for comprehensive verification
  - Stream A: Spectro-Temporal Analysis (CNN-based)
  - Stream B: Acoustic & Biometric Analysis (RNN-based) 
  - Stream C: Linguistic & Prosodic Analysis (Transformer-based)

- **Fusion Engine**: Meta-learner using gradient boosting to make final decisions

- **Detailed Reports**: Generates comprehensive analysis with confidence scores and evidence

- **Intuitive GUI**: User-friendly interface for file selection, recording, and result visualization

- **CLI Support**: Command-line interface for batch processing or integration into other systems

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/APVDS.git
   cd APVDS
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### GUI Mode

Launch the GUI application:

```
python -m ares --gui
```

### Command Line Mode

Analyze an audio file:

```
python -m ares --input /path/to/audio.wav
```

### Python API

```python
from ares.core.engine import ARESEngine

# Initialize the engine
engine = ARESEngine()

# Analyze a file
result = engine.analyze_file("path/to/audio.wav")

# Print the result
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence_score']:.2f}%")
```

## How It Works

ARES employs a multi-layered analysis approach:

1. **Pre-processing**: Audio is cleaned, standardized, and normalized
2. **Parallel Analysis**: Three specialized engines analyze different aspects of the audio
3. **Fusion**: Results are intelligently combined to produce a final verdict
4. **Reporting**: Detailed analysis showing which factors contributed to the decision

## System Requirements

- Python 3.8+
- TensorFlow 2.6.0+
- PyTorch 1.9.0+
- Sufficient RAM for model loading (4GB+ recommended)
- GPU recommended for faster analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This system was inspired by research in audio forensics and deep learning approaches to deepfake detection
- Thanks to the many open source libraries that made this project possible 