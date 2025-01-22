# Make sure we're in the right directory
pwd

# Remove any existing git configuration (if needed)
rm -rf .git

# Initialize a fresh git repository
git init

# Create and switch to main branch
git checkout -b main

# Add all your project files
git add README.md requirements.txt project_structure.sh src/main.py .gitignore

# Create the initial commit
git commit -m "Initial commit: AnimateDiff Temporal Coherence project structure"

# Set the remote origin (if not already set)
git remote add origin https://github.com/Rishardmunene/Motion-Synthesis.git

# Force push to main
git push -f origin main 