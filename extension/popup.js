document.getElementById("scanButton").addEventListener("click", () => {
  const input = document.getElementById("imageInput");
  const resultDiv = document.getElementById("result");

  if (!input.files || input.files.length === 0) {
    resultDiv.textContent = "Please select an image file.";
    return;
  }

  const file = input.files[0];
  const formData = new FormData();
  formData.append("image", file);

  resultDiv.textContent = "Scanning...";

  fetch("http://127.0.0.1:5000/scan_image", {
    method: "POST",
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      resultDiv.textContent = `Error: ${data.error}`;
    } else {
      resultDiv.textContent = `Label: ${data.label}\nConfidence: ${data.confidence}%`;
    }
  })
  .catch(error => {
    resultDiv.textContent = `Request failed: ${error}`;
  });
});
