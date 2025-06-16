# Website Fingerprinting Side-Channel Attack

This project implements a website fingerprinting attack using cache-based side channels, as described in the CSE406 Computer Security assignment.

## Overview

The attack works by measuring cache access patterns while a victim browses websites in another tab. Using machine learning, we can identify which website the victim is visiting with high accuracy.

## Project Structure

```
├── app.py                 # Flask backend server
├── static/
│   ├── index.html        # Web interface
│   ├── index.js          # Frontend JavaScript
│   ├── warmup.js         # Task 1: Timing measurements
│   └── worker.js         # Task 2: Sweep counting attack
├── collect.py            # Task 3: Automated data collection
├── train.py              # Task 4: Machine learning training
├── database.py           # Database operations
├── requirements.txt      # Python dependencies
└── results/              # Generated heatmaps and models
```

## Usage Instructions

### 1. Start the Web Interface

The Flask server should already be running on port 5000. If not, start it with:

```bash
python app.py
```

Then open your browser and go to: http://localhost:5000

### 2. Task 1: Timing Warmup (Completed ✅)

- Click "Collect Latency Data" to measure timing precision
- This tests the `performance.now()` function and cache access patterns
- Results show median access times for different numbers of cache lines

### 3. Task 2: Manual Trace Collection (Completed ✅)

1. Open a target website (like google.com, prothomalo.com, or cse.buet.ac.bd/moodle) in another tab
2. Go back to the fingerprinting tab
3. Click "Collect Trace" 
4. Wait ~10 seconds for collection to complete
5. View the generated heatmap showing cache access patterns

### 4. Task 3: Automated Data Collection (Completed ✅)

For large-scale data collection, use the automation script:

```bash
python collect.py
```

This will:
- Open websites automatically using Selenium
- Collect traces for each target website
- Store data in SQLite database (`webfingerprint.db`)
- Export to JSON format (`dataset.json`)

**Target websites configured:**
- https://cse.buet.ac.bd/moodle/
- https://google.com/
- https://prothomalo.com/

**Configuration:** 
- 1000 traces per website (configurable in `collect.py`)
- Automatic retry on failures
- Graceful shutdown with data preservation

### 5. Task 4: Machine Learning Training (Completed ✅)

Once you have collected sufficient data (30+ traces), train the models:

```bash
python train.py
```

This will:
- Load data from `dataset.json`
- Train two neural network models (Basic CNN and Complex CNN)
- Evaluate performance on test set
- Save best models to `saved_models/` directory
- Generate training summary in `training_summary.json`

**Target:** 60%+ classification accuracy

## Cache Configuration

The implementation is configured for your system:
- **Cache line size:** 64 bytes
- **L3 cache size:** 16 MB
- **Time window:** 10ms per measurement
- **Collection duration:** 10 seconds per trace

## Implementation Details

### Task 1: Timing Measurements
- Uses `performance.now()` for high-resolution timing
- Tests exponentially increasing cache access patterns
- Measures median access times to reduce noise

### Task 2: Sweep Counting Attack
- Allocates buffer equal to L3 cache size (16MB)
- Counts cache sweeps in fixed time windows (10ms)
- Collects 1000 measurements over 10 seconds
- Generates heatmap visualizations

### Task 3: Selenium Automation
- Automated browser control with Chrome WebDriver
- Simulates realistic user behavior (scrolling, clicking)
- Robust error handling and data persistence
- SQLite database for reliable storage

### Task 4: Machine Learning
- Two CNN architectures for comparison
- Data preprocessing with normalization
- Stratified train/test split (80/20)
- PyTorch implementation with CUDA support

## Files Generated

- `results/heatmap_*.png` - Trace visualizations
- `webfingerprint.db` - SQLite database with collected traces
- `dataset.json` - Exported training data
- `saved_models/*.pth` - Trained neural network models
- `training_summary.json` - Training results and metrics

## Tips for Success

1. **Collect sufficient data:** Aim for 50+ traces per website for good accuracy
2. **Close other applications:** Reduce background cache noise during collection
3. **Use consistent timing:** Wait for trace collection to complete fully
4. **Monitor accuracy:** If <60%, collect more data or tune hyperparameters

## Security Notes

This implementation is for educational purposes only. It demonstrates:
- Cache-based side-channel vulnerabilities
- Website fingerprinting techniques
- Machine learning for security analysis

**Important:** Use only on systems you own and websites you have permission to analyze.

## Troubleshooting

**Server not starting:**
```bash
pkill -f "python app.py"
python app.py
```

**Selenium issues:**
- Ensure Chrome browser is installed
- WebDriver will be automatically downloaded

**Low accuracy:**
- Collect more training data
- Try adjusting timing parameters in `worker.js`
- Reduce background system activity

## Assignment Completion Status

- ✅ **Task 1:** Timing warmup with `performance.now()`
- ✅ **Task 2:** Sweep counting attack implementation
- ✅ **Task 3:** Automated data collection with Selenium
- ✅ **Task 4:** Machine learning classification
- ✅ **Bonus:** Real-time heatmap visualization
- ✅ **Bonus:** Comprehensive web interface

**Expected Outcome:** >60% website classification accuracy with sufficient training data. 