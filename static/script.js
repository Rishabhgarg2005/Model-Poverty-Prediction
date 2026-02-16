// ===== Prediction Form Handler =====
document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("prediction-form");
    const btnText = document.querySelector(".btn-text");
    const btnLoader = document.querySelector(".btn-loader");
    const placeholder = document.getElementById("result-placeholder");
    const resultBody = document.getElementById("result-body");
    const resultError = document.getElementById("result-error");

    if (!form) return;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        // Collect form data
        const formData = {};
        const elements = form.elements;
        for (let el of elements) {
            if (el.name) {
                formData[el.name] = el.value;
            }
        }

        // UI: show loading state
        btnText.classList.add("hidden");
        btnLoader.classList.remove("hidden");
        placeholder.classList.add("hidden");
        resultBody.classList.add("hidden");
        resultError.classList.add("hidden");

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(formData),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Prediction failed.");
            }

            showResult(data);
        } catch (err) {
            showError(err.message);
        } finally {
            btnText.classList.remove("hidden");
            btnLoader.classList.add("hidden");
        }
    });

    function showResult(data) {
        const { prediction, classification } = data;

        // Expenditure value
        const expEl = document.getElementById("result-expenditure");
        expEl.textContent = `$${prediction.toFixed(2)}`;
        expEl.style.color = classification.color;

        // Badge
        const badge = document.getElementById("result-badge");
        badge.style.background = hexToRgba(classification.color, 0.15);
        badge.style.color = classification.color;
        document.getElementById("badge-icon").textContent = classification.icon;
        document.getElementById("badge-text").textContent = classification.level;

        // Description
        document.getElementById("result-desc").textContent = classification.desc;

        // Gauge marker position (map prediction to 0-100%)
        const maxVal = 20; // cap for gauge display
        const pct = Math.min(Math.max(prediction / maxVal, 0), 1) * 100;
        document.getElementById("gauge-marker").style.left = pct + "%";

        // Show result
        resultBody.classList.remove("hidden");
        resultError.classList.add("hidden");

        // Scroll to result on mobile
        if (window.innerWidth <= 900) {
            document.getElementById("result-panel").scrollIntoView({ behavior: "smooth" });
        }
    }

    function showError(msg) {
        document.getElementById("error-text").textContent = msg;
        resultError.classList.remove("hidden");
        resultBody.classList.add("hidden");
    }

    function hexToRgba(hex, alpha) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r},${g},${b},${alpha})`;
    }
});
