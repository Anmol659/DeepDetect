uploadForm.addEventListener('submit', (e) => {
  e.preventDefault();

  const file = fileInput.files[0];
  if (!file) {
    resultDiv.textContent = "Please select a file.";
    return;
  }

  const formData = new FormData();
  formData.append('file', file);

  resultDiv.textContent = "Analyzing... Please wait.";

  fetch('/analyze', {
    method: "POST",
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    let color = "black";
    if (data.label === "real") color = "green";
    else if (data.label === "possibly fake") color = "orange";
    else if (data.label === "fake") color = "red";

    resultDiv.innerHTML = `
      <strong style="color:${color}">Result: ${data.label.toUpperCase()}</strong><br/>
      AI Generated: ${(data.probabilities.ai_generated * 100).toFixed(1)}%<br/>
      Deepfake: ${(data.probabilities.deepfake * 100).toFixed(1)}%<br/>
      Real: ${(data.probabilities.real * 100).toFixed(1)}%
    `;
  })
  .catch(error => {
    console.error(error);
    resultDiv.textContent = "Error analyzing file.";
  });
});
