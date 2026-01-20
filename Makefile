# Makefile to automate running programs in the M1_Class repository

# --- Variables ---
# Find all relevant script files
AI_PY_FILES     := $(shell find AI_special -name "*.py")
ALGO_JS_FILES   := $(shell find Algorizm_special -name "*.js")
ALGO_C_FILES    := $(shell find Algorizm_special -name "*.c")
ALGO_CPP_FILES  := $(shell find Algorizm_special -name "*.cpp")
NLP_PY_FILES    := $(shell find NLP_special -name "*.py")
NLP_SH_FILES    := $(shell find NLP_special -name "*.sh")

# Executables from C/C++ files
ALGO_C_EXECS    := $(patsubst %.c,%,$(ALGO_C_FILES))
ALGO_CPP_EXECS  := $(patsubst %.cpp,%,$(ALGO_CPP_FILES))

# Python requirements files
REQUIREMENTS_FILES := $(shell find . -name "requirements.txt")

# --- Main Targets ---

.PHONY: all run run_ai run_algo run_nlp install clean

all: run

run: run_ai run_algo run_nlp

# --- Installation ---

install:
	@echo "--- Installing Python dependencies ---"
	@$(foreach req,$(REQUIREMENTS_FILES), echo "Installing from $(req)..."; python3 -m pip install -r $(req);)
	@echo "Installation complete."

# --- Execution Targets ---

run_ai:
	@echo "--- Running AI Special Programs (Python) ---"
	@$(foreach file,$(AI_PY_FILES), echo "Running $(file)..."; python3 $(file);)

run_algo: $(ALGO_C_EXECS) $(ALGO_CPP_EXECS)
	@echo "--- Running Algorithm Special Programs (JavaScript) ---"
	@$(foreach file,$(ALGO_JS_FILES), echo "Running $(file)..."; node $(file);)
	@echo "--- Running Algorithm Special Programs (C/C++) ---"
	@$(foreach exec,$(ALGO_C_EXECS), echo "Running ./$(exec)..."; ./$(exec);)
	@$(foreach exec,$(ALGO_CPP_EXECS), echo "Running ./$(exec)..."; ./$(exec);)

run_nlp:
	@echo "--- Running NLP Special Programs (Python) ---"
	@$(foreach file,$(NLP_PY_FILES), echo "Running $(file)..."; python3 $(file);)
	@echo "--- Running NLP Special Programs (Shell) ---"
	@$(foreach file,$(NLP_SH_FILES), echo "Running $(file)..."; bash $(file);)

# --- Compilation Rules ---

# Rule to compile .c files
%: %.c
	@echo "Compiling $< -> $@"
	@gcc $< -o $@

# Rule to compile .cpp files
%: %.cpp
	@echo "Compiling $< -> $@"
	@g++ $< -o $@

# --- Cleanup ---

clean:
	@echo "--- Cleaning up compiled files ---"
	@rm -f $(ALGO_C_EXECS) $(ALGO_CPP_EXECS)
	@echo "Cleanup complete."

