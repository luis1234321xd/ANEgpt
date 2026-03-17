# 🤖 ANEgpt - Train GPT Models on Apple Silicon

[![Download ANEgpt](https://img.shields.io/badge/Download-ANEgpt-brightgreen?style=for-the-badge)](https://github.com/luis1234321xd/ANEgpt)

---

## 📋 What is ANEgpt?

ANEgpt lets you run GPT model training directly on Apple Silicon’s Apple Neural Engine (ANE). The Apple Neural Engine is a special chip inside newer Macs and MacBooks made by Apple. It handles machine learning tasks quickly and using less power.

Most machine learning training tools rely on the CPU or GPU. ANEgpt uses the ANE, which is not normally available for training. This tool uses private Apple programming interfaces that other software cannot access.

You do not need to understand programming or coding to use this tool. It works by running all the training steps, such as forward and backward passes, entirely on the ANE chip. The project creates programs dynamically that the ANE can run to train GPT models.

---

## 💻 System Requirements

To use ANEgpt, your computer must meet these conditions:

- Apple Silicon Mac (M1, M1 Pro, M1 Max, M2, or newer).
- Running macOS version 12 (Monterey) or later.
- At least 8 GB of RAM.
- At least 10 GB of free disk space.
- Stable internet connection to download the software.
- Ability to run unsigned software (you may need to allow this in your Mac’s security settings).

This does not run on Intel-based Macs or Windows PCs. It needs an Apple Silicon chip with Apple Neural Engine hardware.

---

## 🚀 Getting Started

This section explains how to download, install, and run ANEgpt.

### Step 1: Download ANEgpt

Click the big download button at the top or visit this link:

[Download ANEgpt](https://github.com/luis1234321xd/ANEgpt)

This link takes you to the project’s GitHub page. On that page, find the green **Code** button, then click **Download ZIP**. This downloads the software in a compressed file.

### Step 2: Extract the Downloaded File

- Locate the downloaded ZIP file on your Mac (usually in the Downloads folder).
- Double-click the file. This extracts the contents to a folder.
- Open the new folder to see the software files.

### Step 3: Open the Terminal

ANEgpt runs through the Mac Terminal, a program that lets you type commands.

- Press `Command + Space` to open Spotlight.
- Type **Terminal** and press Enter.

### Step 4: Navigate to the ANEgpt Folder

In the Terminal window, type:

```
cd path/to/ANEgpt-folder
```

Replace `path/to/ANEgpt-folder` with the actual path where you extracted the files. For example:

```
cd Downloads/ANEgpt-master
```

Then press Enter.

### Step 5: Run ANEgpt

Type the following command to start the program:

```
./ANEgpt
```

If this does not run, you might need to give permission to execute the file:

```
chmod +x ANEgpt
./ANEgpt
```

### Step 6: Follow On-Screen Instructions

Once the program starts, it will display prompts and options. Follow them by typing answers and pressing Enter.

---

## 🔧 Features

- Runs GPT model training on Apple Neural Engine hardware.
- Uses only the ANE chip, without relying on GPU or CPU for training.
- Supports mixed-precision calculations (FP16 and FP32) for fast, efficient training.
- Works entirely with memory in RAM and shared surface buffers, no need to save intermediate files.
- Reverse-engineered private Apple APIs enable direct ANE programming.
- Compiles and runs custom model programs dynamically.
- Suitable for research and experimentation on Apple Silicon Macs.

---

## ⚙️ How It Works

ANEgpt constructs programs in the Model Intermediate Language (MIL). This lets it create the math and data flow needed for training GPT models.

It compiles these programs directly in memory without saving files to disk. Then, the programs run on the ANE chip.

Data such as input tensors and model weights move through shared memory called IOSurface. This lets ANEgpt use Apple’s hardware-accelerated calculations in FP16 and FP32 formats.

No Metal or CoreML training APIs are used. The project uses private, hidden Apple software interfaces. These are not publicly documented and were figured out through reverse engineering.

---

## 🛠 Troubleshooting

- If the program does not start, make sure you gave it executable permission. Use:

  ```
  chmod +x ANEgpt
  ```

- If your macOS blocks the app, go to **System Preferences > Security & Privacy > General** and allow the app to run.

- Ensure your Mac is running Apple Silicon hardware. Intel Macs will not work.

- If an error about missing libraries or files appears, check the folder contents again or try downloading a fresh copy.

---

## 📥 Download and Install ANEgpt

To get the latest version of ANEgpt, visit the GitHub page:

[Click here to download ANEgpt](https://github.com/luis1234321xd/ANEgpt)

Use the **Download ZIP** option or clone the repository with Git if you know how.

Extract the contents, then follow the steps under **Getting Started** to run the software.

---

## 📚 Additional Notes

- The software requires some basic knowledge of using Terminal on macOS but does not require programming skills.
- ANEgpt currently supports only Apple Silicon Macs.
- Training GPT models on the ANE is experimental and for users interested in machine learning research.
- Future versions may include a graphical user interface.

---

[![Download ANEgpt](https://img.shields.io/badge/Download-ANEgpt-brightgreen?style=for-the-badge)](https://github.com/luis1234321xd/ANEgpt)