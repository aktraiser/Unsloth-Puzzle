Unsloth Puzzle Solutions
This repository contains my solutions for the Unsloth coding puzzles, which serve as a technical challenge for potential engineering roles at Unsloth. The solutions here demonstrate my work on optimizing kernels, enabling QLoRA with FSDP2, making torch.compile work without graph breaks, and other AI/ML-related tasks.

🚀 Project Overview
This repository covers the following puzzles:

Puzzle A: Convert nf4 to a Triton kernel.
Puzzle B: Implement QLoRA with FSDP2 on multiple GPUs.
Puzzle C: Optimize torch.compile to remove graph breaks in QLoRA.
Puzzle D: Solve Unsloth GitHub issues (e.g., Flex Attention, VLMs support, GGUF Vision support).
Puzzle E: Implement memory-efficient backpropagation for large vocabulary LLMs.
The repository is structured for easy navigation, testing, and submission.

📂 Repository Structure
my-unsloth-solutions/
├── README.md  # This file
├── environment.yml  # Conda environment file (optional)
├── requirements.txt  # Python dependencies
├── notebooks/
│    ├── Unsloth_Puzzles.ipynb  # (Optional) Original challenge notebook
│    ├── puzzle_A_triton_dequant.ipynb
│    ├── puzzle_B_fsdp2_qlora.ipynb
│    ├── puzzle_C_torch_compile.ipynb
│    ├── puzzle_D_unsloth_issues.md
│    └── puzzle_E_mem_efficient_backprop.ipynb
└── src/
     ├── my_triton_kernel.py  # Triton kernel for Puzzle A
     ├── my_fsdp2_script.py  # Training script for Puzzle B
     └── ...

     
🛠 Installation
Local Setup
To run the notebooks locally, clone the repository and install the necessary dependencies:

git clone https://github.com/YOUR_GITHUB_USERNAME/my-unsloth-solutions.git
cd my-unsloth-solutions
pip install -r requirements.txt
Alternatively, if using Conda:

conda env create -f environment.yml
conda activate unsloth-env
Ensure you have a GPU with CUDA 11.4+ (Tesla T4 recommended).

📊 Running the Solutions
Each puzzle has its own notebook/script:

Puzzle A: Convert nf4 to Triton
cd notebooks
jupyter notebook puzzle_A_triton_dequant.ipynb
Run test_dequantize(your_dequantize_nf4) to check performance.
Verify speedup (test_dequantize(unsloth_dequantize) / test_dequantize(your_dequantize_nf4)).
Puzzle B: QLoRA with FSDP2
Run the Kaggle notebook (link below) to showcase multi-GPU finetuning on 2x Tesla T4 GPUs.
Ensure the loss matches single-GPU results.
Puzzle C: Optimize torch.compile
Ensure there are no graph breaks (TORCHDYNAMO_VERBOSE=1).
Show the training loss curve to validate correctness.
Puzzle D: Solve Unsloth GitHub Issues
Links to pull requests and implemented features are included in puzzle_D_unsloth_issues.md.
Puzzle E: Memory-Efficient Backpropagation
Implemented using torch.autograd.Function.
Should reduce VRAM usage while keeping gradients numerically equivalent.
