document.addEventListener("DOMContentLoaded", () => {
  function sessionValue(key) {
    try {
      return window.sessionStorage?.getItem(key) || "";
    } catch (_error) {
      return "";
    }
  }

  const geminiKey = sessionValue("jobpilotGeminiApiKey");
  const geminiModel = sessionValue("jobpilotGeminiModel");

  document.querySelectorAll(".resume-form").forEach((form) => {
    const button = form.querySelector(".resume-button");
    const keyInput = form.querySelector('input[name="resume_api_key"]');
    const modelInput = form.querySelector('input[name="resume_model"]');
    const serverAvailable = form.dataset.serverAvailable === "1";

    if (keyInput) keyInput.value = geminiKey;
    if (modelInput) modelInput.value = geminiModel;

    if (button && (serverAvailable || geminiKey)) {
      button.disabled = false;
      button.removeAttribute("title");
      button.textContent = "Generate Resume";
    }

    form.addEventListener("submit", (event) => {
      if (!serverAvailable && !geminiKey) {
        event.preventDefault();
        if (button) {
          button.disabled = true;
          button.title = "Read the profile with a Gemini API key first, then generate a resume.";
        }
        return;
      }
      if (button) {
        button.disabled = true;
        button.textContent = "Generating...";
      }
    });
  });
});
