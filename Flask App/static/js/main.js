/**
 * Fashion AI Classifier - Main JavaScript
 * Handles image upload, prediction and UI interactions
 */

document.addEventListener("DOMContentLoaded", function () {
  // DOM Elements
  const dropArea = document.getElementById("drop-area");
  const fileInput = document.getElementById("file-input");
  const previewContainer = document.getElementById("preview-container");
  const imagePreview = document.getElementById("image-preview");
  const removeImageBtn = document.getElementById("remove-image");
  const uploadBtn = document.getElementById("upload-btn");
  const resultsSection = document.getElementById("results-section");
  const placeholderSection = document.getElementById("placeholder-section");
  const errorSection = document.getElementById("error-section");
  const errorMessage = document.getElementById("error-message");
  const errorDismiss = document.getElementById("error-dismiss");
  const predictionLabel = document.getElementById("prediction-label");
  const confidenceBar = document.getElementById("confidence-bar");
  const confidenceValue = document.getElementById("confidence-value");
  const allCategories = document.getElementById("all-categories");
  const categoryIcon = document.getElementById("category-icon");
  const loadingOverlay = document.getElementById("loading-overlay");

  // Track current image file
  let currentFile = null;

  // Event Listeners for drag and drop
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ["dragenter", "dragover"].forEach((eventName) => {
    dropArea.addEventListener(eventName, highlight, false);
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, unhighlight, false);
  });

  function highlight() {
    dropArea.classList.add("dragover");
  }

  function unhighlight() {
    dropArea.classList.remove("dragover");
  }

  // Handle dropped files
  dropArea.addEventListener("drop", handleDrop, false);

  function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;

    if (files.length) {
      handleFiles(files);
    }
  }

  // Handle file selection via input
  fileInput.addEventListener("change", function () {
    if (this.files.length) {
      handleFiles(this.files);
    }
  });

  // Process selected files
  function handleFiles(files) {
    const file = files[0];

    // Check if file is an image
    if (!file.type.match("image.*")) {
      showError("Please select a valid image file (JPEG, PNG, etc.)");
      return;
    }

    currentFile = file;
    displayImagePreview(file);
    uploadBtn.disabled = false;
  }

  // Display image preview
  function displayImagePreview(file) {
    const reader = new FileReader();

    reader.onload = function (e) {
      imagePreview.src = e.target.result;
      previewContainer.classList.remove("d-none");
      dropArea.querySelector(".drop-message").classList.add("d-none");

      // Add subtle animation
      imagePreview.style.opacity = 0;
      setTimeout(() => {
        imagePreview.style.opacity = 1;
      }, 50);
    };

    reader.readAsDataURL(file);
  }

  // Remove preview image
  removeImageBtn.addEventListener("click", function () {
    resetImageUpload();
  });

  function resetImageUpload() {
    currentFile = null;
    previewContainer.classList.add("d-none");
    dropArea.querySelector(".drop-message").classList.remove("d-none");
    uploadBtn.disabled = true;
    fileInput.value = "";
  }

  // Handle image upload and prediction
  uploadBtn.addEventListener("click", function () {
    if (!currentFile) return;

    // Show loading overlay
    loadingOverlay.classList.remove("d-none");

    // Hide any previous results or errors
    resultsSection.classList.add("d-none");
    placeholderSection.classList.add("d-none");
    errorSection.classList.add("d-none");

    // Create form data
    const formData = new FormData();
    formData.append("file", currentFile);

    // Send to server
    fetch("/predict", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        loadingOverlay.classList.add("d-none");

        if (data.error) {
          showError(data.error);
          return;
        }

        // Display results
        displayResults(data);
      })
      .catch((error) => {
        loadingOverlay.classList.add("d-none");
        showError(`Error: ${error.message}`);
      });
  });

  // Display prediction results
  function displayResults(data) {
    // Show results section
    resultsSection.classList.remove("d-none");
    placeholderSection.classList.add("d-none");
    errorSection.classList.add("d-none");

    // Update prediction label with animation
    predictionLabel.textContent = capitalizeFirst(data.prediction);
    predictionLabel.style.animation = "none";
    setTimeout(() => {
      predictionLabel.style.animation = "fadeIn 0.8s ease";
    }, 10);

    // Update confidence bar
    const confidencePercent = Math.round(data.confidence * 100);
    confidenceBar.style.width = `${confidencePercent}%`;
    confidenceValue.textContent = `${confidencePercent}%`;

    // Change color based on confidence
    if (confidencePercent >= 0) {
      confidenceBar.style.backgroundColor = "var(--success-color)";
      // Trigger confetti for high confidence
      launchConfetti();
    } else if (confidencePercent >= 50) {
      confidenceBar.style.backgroundColor = "var(--primary-color)";
    } else {
      confidenceBar.style.backgroundColor = "var(--warning-color)";
    }

    // Set appropriate icon for prediction
    setCategoryIcon(data.prediction);

    // Display all confidence scores
    displayAllConfidences(data.all_confidences, data.prediction);

    // Show a small badge if we're in demo mode
    if (data.demo_mode) {
      // Check if demo badge already exists
      if (!document.getElementById("demo-badge")) {
        const demoBadge = document.createElement("div");
        demoBadge.id = "demo-badge";
        demoBadge.className = "demo-badge";
        demoBadge.innerHTML = "Demo Mode";

        // Add a tooltip explaining what demo mode is
        demoBadge.title =
          "Currently using random predictions for demonstration purposes";

        // Add it to the results section
        document.querySelector(".result-header").appendChild(demoBadge);

        // Add some CSS for the demo badge
        const style = document.createElement("style");
        style.textContent = `
                    .demo-badge {
                        background-color: var(--secondary-color);
                        color: white;
                        padding: 4px 8px;
                        border-radius: 4px;
                        font-size: 0.7rem;
                        font-weight: 500;
                        position: absolute;
                        top: 0;
                        right: 0;
                        opacity: 0.8;
                        cursor: help;
                    }
                `;
        document.head.appendChild(style);
      }
    }
  }

  // Set category icon based on prediction
  function setCategoryIcon(category) {
    // Reset classes
    categoryIcon.className = "";

    // Set appropriate icon for each category
    switch (category.toLowerCase()) {
      case "pants":
        categoryIcon.className = "fas fa-ruler-vertical"; // Changed from fa-socks to a more pants-like icon
        break;
      case "long sleeve":
        categoryIcon.className = "fas fa-tshirt";
        break;
      case "dress":
        categoryIcon.className = "fas fa-female";
        break;
      case "bags":
        categoryIcon.className = "fas fa-shopping-bag";
        break;
      case "footwear":
        categoryIcon.className = "fas fa-shoe-prints";
        break;
      default:
        categoryIcon.className = "fas fa-tshirt";
    }
  }

  // Display all confidence scores
  function displayAllConfidences(confidences, topCategory) {
    // Clear previous results
    allCategories.innerHTML = "";

    // Sort categories by confidence score
    const sortedCategories = Object.entries(confidences).sort(
      (a, b) => b[1] - a[1]
    );

    // Create bar for each category
    sortedCategories.forEach(([category, confidence], index) => {
      const confidencePercent = Math.round(confidence * 100);
      const isHighest = category.toLowerCase() === topCategory.toLowerCase();

      const categoryBar = document.createElement("div");
      categoryBar.className = "category-bar";
      categoryBar.style.animationDelay = `${0.1 * (index + 1)}s`;

      categoryBar.innerHTML = `
                <div class="category-label">
                    <span class="name">${capitalizeFirst(category)}</span>
                    <span class="value">${confidencePercent}%</span>
                </div>
                <div class="category-progress">
                    <div class="category-progress-bar ${
                      isHighest ? "highest" : ""
                    }" 
                         style="width: 0%"></div>
                </div>
            `;

      allCategories.appendChild(categoryBar);

      // Animate progress bar after a small delay
      setTimeout(() => {
        const progressBar = categoryBar.querySelector(".category-progress-bar");
        progressBar.style.width = `${confidencePercent}%`;
      }, 100);
    });
  }

  // Show error message
  function showError(message) {
    resultsSection.classList.add("d-none");
    placeholderSection.classList.add("d-none");
    errorSection.classList.remove("d-none");

    errorMessage.textContent = message;
  }

  // Dismiss error
  errorDismiss.addEventListener("click", function () {
    errorSection.classList.add("d-none");
    placeholderSection.classList.remove("d-none");
  });

  // Helper function to capitalize first letter
  function capitalizeFirst(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
  }

  // Launch confetti for successful high-confidence prediction
  function launchConfetti() {
    if (typeof confetti !== "undefined") {
      confetti.start();

      // Stop after a shorter time (1.5 seconds)
      setTimeout(() => {
        confetti.stop();

        // Clear any remaining confetti particles after they've fallen
        setTimeout(() => {
          confetti.clear();
        }, 1000);
      }, 600);
    }
  }
});
