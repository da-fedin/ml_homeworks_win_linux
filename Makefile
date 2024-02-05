# Set variables
VENV_DIR := venv
SCRIPTS_DIR := scripts
PS_SCRIPT := $(SCRIPTS_DIR)/build_project.ps1
REQUIREMENTS_FILE := requirements.txt

# Build project
build-project:
	@powershell -ExecutionPolicy Bypass -File $(PS_SCRIPT)

# Install pre-commit hook
pre-commit-install:
	.venv\Scripts\pre-commit.exe install
