# How to Run the Streamlit App

## Quick Start

1. **Install dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

2. **Train models first** (required before using the app):
   ```bash
   python train_models.py
   ```
   This creates the `models/` directory with pretrained models.

3. **Run the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

   Or alternatively:
   ```bash
   python -m streamlit run streamlit_app.py
   ```

4. **Access the app**:
   - The app will automatically open in your default web browser
   - Usually at: `http://localhost:8501`
   - If it doesn't open automatically, copy the URL from the terminal

## Common Errors and Solutions

### Error 1: "ModuleNotFoundError: No module named 'streamlit'"
**Solution:**
```bash
pip install streamlit
```

### Error 2: "Pretrained model not found"
**Error Message:** `❌ Pretrained model 'combined' not found`

**Solution:**
1. Make sure you have dataset files in the `data/` directory
2. Run: `python train_models.py`
3. Check that `models/` directory exists with `.pkl` files

### Error 3: "No module named 'dataset_classes'"
**Solution:**
- Make sure you're running the command from the project root directory (`d:\projects\eeg`)
- All Python files should be in the same directory

### Error 4: "FileNotFoundError" when uploading
**Solution:**
- Make sure your uploaded file is a valid MATLAB `.mat` file
- Check that the file format matches BCIC4 or OpenBMI format

### Error 5: "CSP matrix dimension mismatch"
**Error:** This happens when the uploaded dataset has different number of channels than the training data

**Solution:**
- Use a model trained on similar data (e.g., use `bcic4` model for BCIC4 data)
- Or retrain models with your specific dataset

### Error 6: Streamlit port already in use
**Error:** `Port 8501 is already in use`

**Solution:**
```bash
# Option 1: Use a different port
streamlit run streamlit_app.py --server.port 8502

# Option 2: Kill the process using port 8501
# On Windows:
netstat -ano | findstr :8501
taskkill /PID <PID_NUMBER> /F
```

## Step-by-Step Troubleshooting

### Check Installation
```bash
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
python -c "import numpy, scipy, sklearn, matplotlib, pandas; print('All packages OK')"
```

### Check Imports
```bash
python -c "from dataset_classes import MotorImageryBcic4, MotorImageryOpenBmi; print('Imports OK')"
```

### Check Models Directory
```bash
# On Windows PowerShell:
dir models\*.pkl

# Should show:
# models/bcic4_model.pkl
# models/bcic4_csp.pkl
# models/openbmi_model.pkl
# models/openbmi_csp.pkl
# models/combined_model.pkl
# models/combined_csp.pkl
```

## Running in Different Environments

### Windows PowerShell
```powershell
cd d:\projects\eeg
streamlit run streamlit_app.py
```

### Windows Command Prompt
```cmd
cd d:\projects\eeg
streamlit run streamlit_app.py
```

### With Virtual Environment
```bash
# Activate venv first
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Then run
streamlit run streamlit_app.py
```

## App Features Checklist

Before using the app, ensure:
- ✅ All dependencies installed (`pip install -r requirements.txt`)
- ✅ Models trained (`python train_models.py`)
- ✅ Dataset files available in `data/` directory (for training)
- ✅ You have a `.mat` file to upload (for classification)

## Getting Help

If you encounter errors:
1. Check the terminal/console output for detailed error messages
2. Verify all files are in the correct directory
3. Make sure Python version is 3.7 or higher
4. Check that all required packages are installed

