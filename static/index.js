document.getElementById("predict-form").addEventListener("submit", function(e) {
    e.preventDefault();

    const formData = new FormData(this);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const resultSection = document.getElementById("prediction-result");
        const resultText = document.getElementById("result-text");

        resultText.innerText = data.result;
        resultSection.style.display = "block";
    })
    .catch(error => {
        console.error("Error:", error);
        const resultText = document.getElementById("result-text");
        resultText.innerText = "Something went wrong.";
        document.getElementById("prediction-result").style.display = "block";
    });
});
