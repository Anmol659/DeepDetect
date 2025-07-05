// content.js
document.querySelectorAll("img").forEach(img => {
  // For example: add a border to show we're scanning
  img.style.border = "2px solid blue";

  // You'd fetch the image data here, send to background.js
  // then get the prediction and overlay it on the image
});
