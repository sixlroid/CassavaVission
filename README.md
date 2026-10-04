# CassavaVision — Cassava Leaf Disease Diagnostic System

---

## 🛠️ Setup & Installation

### 1. To download the repository (Clone)
If you want to copy this repository from GitHub to your local machine:
```bash
git clone https://github.com/six1roid/CassavaVission.git
cd CassavaVission
```

### 2. To install dependencies (Streamlit & Ultralytics)
If you need to install or update the required Python libraries locally:
```bash
pip install -r requirements.txt
```
*(If you encounter a path launcher error on Windows, use: `python -m pip install -r requirements.txt`)*

### 3. To run the app locally on your browser (Localhost)
To launch the Streamlit web interface on your local machine:
```bash
streamlit run app.py
```
*(Or use: `python -m streamlit run app.py`)*

### 4. To install Streamlit independently
If you only need to install Streamlit without the full requirements file:
```bash
pip install streamlit
```
*(On Windows, if you run into path errors, use: `python -m pip install streamlit`)*

---

## 🚀 Git Workflow Guide

### To sync changes from GitHub (Pull)
If someone else updated the repository or you made changes on another device and want to pull the latest version:
```bash
git pull origin main
```

### To save and log your local changes (Commit)
When you have modified files and want to package them locally with a descriptive message:
```bash
git add .
git commit -m "feat: describe your changes here"
```

### To upload your local commits to GitHub (Push)
When you want to send your saved local commits up to the remote repository:
```bash
git push origin main
```

### To look at or inspect a previous commit (Checkout)
If you want to temporarily view the code as it looked at a specific historical commit without losing current work:
```bash
# 1. View your commit history to find the commit hash (e.g., a1b2c3d)
git log --oneline

# 2. Switch to that commit temporarily
git checkout <commit-hash>

# To return back to your latest work on main:
git checkout main
```
