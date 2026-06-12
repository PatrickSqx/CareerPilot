document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("jobpilot-form");
  const parseButton = document.getElementById("btn-parse");
  const parseLabel = document.getElementById("parse-label");
  const parseLoading = document.getElementById("parse-loading");
  const matchButton = document.getElementById("btn-match");
  const matchLabel = document.getElementById("match-label");
  const matchLoading = document.getElementById("match-loading");
  const matchStatus = document.getElementById("match-status");
  const parseError = document.getElementById("parse-error");
  const parseNotes = document.getElementById("parse-notes");
  const fieldsContainer = document.getElementById("fields-container");
  const liveSummary = document.getElementById("live-summary");
  const personaPreset = document.getElementById("persona-preset");
  const profileText = document.getElementById("profile_text");
  const resumeFile = document.getElementById("resume_pdf");
  const filtersReady = document.getElementById("profile_filters_ready");
  const parserMode = document.getElementById("parser_mode");
  const parserProvider = document.getElementById("parser_provider");
  const parserApiKey = document.getElementById("parser_api_key");
  const parserModel = document.getElementById("parser_model");
  let parseInProgress = false;
  const companyControls = {
    size: document.getElementById("company_size_mode")
  };
  const locationPreset = document.getElementById("location_preference_preset");
  const locationCustom = document.getElementById("location_custom_text");
  const locationAddButton = document.getElementById("location_add_button");

  const locationPresetMap = {
    "": [],
    us_remote: ["United States", "Remote"],
    us_any: ["United States"],
    remote_any: ["Remote"],
    bay_area: ["United States", "Remote", "San Francisco", "Bay Area"],
    nyc: ["United States", "Remote", "New York", "NYC"],
    west_coast: ["United States", "Remote", "Seattle", "San Francisco"],
    research_hubs: ["United States", "Remote", "Seattle", "San Francisco", "New York", "Boston"]
  };

  const demoTexts = {
    aisha: `Background:
Career pivoter with analytics coursework and applied machine learning projects.

Skills:
Python, SQL, pandas, scikit-learn, machine learning, analytics

Target Roles:
Machine Learning Engineer, ML Engineer, Applied Scientist, Data Scientist

Preferences:
Remote or Bay Area roles. Salary minimum $140k. ML-related or research AI roles only.

Dealbreakers:
Defense, military, senior, staff, principal.

Pass Criteria:
No senior/staff/principal roles. No roles requiring more than 4 years. Avoid defense or clearance.`,
    kenji: `Background:
International CS graduate student on OPT with machine learning research experience.

Skills:
Python, PyTorch, TensorFlow, machine learning, deep learning, computer vision, NLP

Target Roles:
Research Scientist, Applied Scientist, Machine Learning Engineer, Data Scientist

Preferences:
US only. Remote, Seattle, San Francisco, or New York. Minimum $120k. Prefer large companies or research labs.

Dealbreakers:
No sponsorship, contract, temporary, unpaid, staff, principal, director, lead.

Pass Criteria:
Needs H-1B sponsorship after OPT. No contract/temp. No senior/staff/principal. No 3+ years.`
,
    marcus: `Background:
MBA student with product analytics, stakeholder reporting, and business intelligence project experience.

Skills:
SQL, Excel, Tableau, Power BI, analytics, statistics, stakeholder analytics

Target Roles:
Business Analyst, Data Analyst, Product Analyst, Analytics Consultant

Preferences:
Remote or New York roles. Prefer analytics, BI, product, or strategy roles. Salary minimum $90k.

Dealbreakers:
Heavy software engineering, senior manager, director, staff, principal, unpaid internship.

Pass Criteria:
Full-time roles only. No senior/staff/principal/director roles. No roles requiring more than 4 years.`,
    priya: `Background:
Software engineer with backend systems, data pipelines, cloud infrastructure, and ML platform project experience.

Skills:
Java, Python, Spark, Kafka, AWS, Kubernetes, Docker, microservices, machine learning

Target Roles:
Machine Learning Engineer, ML Platform Engineer, Data Engineer, Backend Engineer

Preferences:
Remote or New York roles. Prefer cloud, data infrastructure, ML platform, or backend engineering teams.

Dealbreakers:
Frontend-only roles, unpaid roles, director, staff, principal, roles requiring clearance.

Pass Criteria:
Full-time roles only. No staff/principal/director roles. No clearance or defense-heavy roles.`
  };

  function setParseBusy(isBusy) {
    if (!parseButton || !parseLabel || !parseLoading) return;
    parseInProgress = isBusy;
    parseButton.disabled = isBusy;
    if (isBusy) setMatchBusy(false);
    parseLabel.classList.toggle("hidden", isBusy);
    parseLoading.classList.toggle("hidden", !isBusy);
    parseLoading.textContent = parserMode?.value === "llm" ? "Calling Gemini..." : "Reading locally...";
  }

  function setMatchBusy(isBusy) {
    if (!matchButton || !matchLabel || !matchLoading) return;
    matchButton.disabled = isBusy;
    matchLabel.classList.toggle("hidden", isBusy);
    matchLoading.classList.toggle("hidden", !isBusy);
    matchStatus?.classList.toggle("hidden", !isBusy);
  }

  function markFiltersReady() {
    if (filtersReady) filtersReady.value = "1";
  }

  function clearFiltersReady() {
    if (filtersReady) filtersReady.value = "";
  }

  function setMessage(element, messages) {
    if (!element) return;
    const items = Array.isArray(messages) ? messages.filter(Boolean) : [messages].filter(Boolean);
    element.textContent = items.join(" ");
    element.classList.toggle("hidden", items.length === 0);
  }

  function updateParseButtonText() {
    if (!parseLabel || !parseLoading) return;
    const useGemini = parserMode?.value === "llm";
    parseLabel.textContent = useGemini ? "Read with Gemini API" : "Read Profile Locally";
    parseLoading.textContent = useGemini ? "Calling Gemini..." : "Reading locally...";
  }

  function rememberParserApiSettings() {
    const key = parserApiKey?.value || "";
    try {
      if (parserMode?.value === "llm" && key.trim()) {
        window.sessionStorage.setItem("jobpilotGeminiApiKey", key);
        window.sessionStorage.setItem("jobpilotGeminiModel", parserModel?.value || "");
      } else if (!key.trim()) {
        window.sessionStorage.removeItem("jobpilotGeminiApiKey");
        window.sessionStorage.removeItem("jobpilotGeminiModel");
      }
    } catch (_error) {
      // Session storage can be unavailable in hardened browser modes.
    }
  }

  function controlValue(control) {
    if (!control) return "";
    if (control.type === "checkbox") return control.checked;
    return control.value.trim();
  }

  function isEmptyValue(control) {
    const value = controlValue(control);
    return value === "" || value === false || value == null;
  }

  function setControlValue(control, value) {
    if (!control) return;
    if (control.type === "checkbox") {
      control.checked = value === true || value === "true" || value === "on" || value === "1";
      return;
    }
    if (Array.isArray(value)) {
      control.value = value.join(", ");
      return;
    }
    control.value = value == null ? "" : String(value);
  }

  function namedControl(name) {
    const control = form?.elements?.[name];
    const isRadioNodeList = typeof RadioNodeList !== "undefined" && control instanceof RadioNodeList;
    if (!control || isRadioNodeList) return null;
    return control;
  }

  function setNamedValue(name, value, source = "user") {
    const control = namedControl(name);
    if (!control) return;
    setControlValue(control, value);
    updateBadge(control, source);
  }

  function listValue(controlName) {
    const control = namedControl(controlName);
    if (!control || control.type === "checkbox") return [];
    return control.value
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
  }

  function appendListItems(controlName, items) {
    const existing = listValue(controlName);
    const merged = [...existing];
    items.forEach((item) => {
      if (!merged.some((value) => value.toLowerCase() === item.toLowerCase())) {
        merged.push(item);
      }
    });
    setNamedValue(controlName, merged.join(", "), "user");
  }

  function setListPreserving(controlName, managedItems, enabledItems, source = "user") {
    const managed = new Set(managedItems.map((item) => item.toLowerCase()));
    const existing = listValue(controlName).filter((item) => !managed.has(item.toLowerCase()));
    const merged = [...existing];
    enabledItems.forEach((item) => {
      if (!merged.some((value) => value.toLowerCase() === item.toLowerCase())) {
        merged.push(item);
      }
    });
    setNamedValue(controlName, merged.join(", "), source);
  }

  function removeListItems(controlName, items) {
    const removals = new Set(items.map((item) => item.toLowerCase()));
    const filtered = listValue(controlName).filter((item) => !removals.has(item.toLowerCase()));
    setNamedValue(controlName, filtered.join(", "), "user");
  }

  function normalizeLocationItem(value) {
    return value.trim().toLowerCase().replace(/\./g, "").replace(/\s+/g, " ");
  }

  function splitLocationText(value) {
    return String(value || "")
      .split(/[,;\n|]+/)
      .map((item) => item.trim())
      .filter(Boolean);
  }

  function mergeLocationItems(items) {
    const merged = [];
    items.forEach((item) => {
      if (!merged.some((value) => normalizeLocationItem(value) === normalizeLocationItem(item))) {
        merged.push(item);
      }
    });
    return merged;
  }

  function isLocationCustomVisible() {
    return Boolean(locationCustom && !locationCustom.classList.contains("hidden"));
  }

  function setLocationCustomVisible(visible) {
    if (!locationCustom) return;
    locationCustom.classList.toggle("hidden", !visible);
    locationCustom.setAttribute("aria-hidden", visible ? "false" : "true");
    if (locationAddButton) locationAddButton.classList.toggle("hidden", visible);
  }

  function refreshLocationCustomVisibility() {
    const hasCustomValue = Boolean((locationCustom?.value || "").trim());
    const wantsCustom = locationPreset?.value === "custom";
    const wasExpanded = locationCustom?.dataset.expanded === "true";
    setLocationCustomVisible(Boolean(hasCustomValue || wantsCustom || wasExpanded));
  }

  function updateLocationCanonical(source = "user") {
    const presetKey = locationPreset?.value || "";
    const presetItems = presetKey === "custom" ? [] : locationPresetMap[presetKey] || [];
    const customItems = splitLocationText(locationCustom?.value || "");
    const merged = mergeLocationItems([...presetItems, ...customItems]);
    setNamedValue("manual_location_preferences", merged.join(", "), source);
    refreshLocationCustomVisibility();
    updateLiveSummary();
  }

  function syncLocationControlsFromCanonical() {
    if (!locationPreset || !locationCustom) return;
    const canonicalItems = listValue("manual_location_preferences");
    if (canonicalItems.length === 0) {
      locationPreset.value = "";
      locationCustom.value = "";
      locationCustom.dataset.expanded = "";
      refreshLocationCustomVisibility();
      return;
    }

    const canonicalKeys = new Set(canonicalItems.map(normalizeLocationItem));
    const candidates = Object.entries(locationPresetMap)
      .filter(([key, items]) => key && items.length > 0 && items.every((item) => canonicalKeys.has(normalizeLocationItem(item))))
      .sort((a, b) => b[1].length - a[1].length);
    const [presetKey, presetItems] = candidates[0] || ["custom", []];
    const presetKeys = new Set(presetItems.map(normalizeLocationItem));
    const customItems = canonicalItems.filter((item) => !presetKeys.has(normalizeLocationItem(item)));
    locationPreset.value = presetKey;
    locationCustom.value = customItems.join(", ");
    locationCustom.dataset.expanded = customItems.length || presetKey === "custom" ? "true" : "";
    refreshLocationCustomVisibility();
  }

  function updateBadge(control, source) {
    const row = control.closest(".filter-row");
    if (!row) return;
    const badge = row.querySelector(".source-badge");
    if (!badge) return;
    const normalized = isEmptyValue(control) ? "empty" : source;
    if (normalized === "user") {
      badge.className = "source-badge user";
      badge.textContent = "User edited";
    } else if (normalized === "inferred") {
      badge.className = "source-badge inferred";
      badge.textContent = "Inferred";
    } else {
      badge.className = "source-badge empty hidden";
      badge.textContent = "";
    }
  }

  function labelFor(control) {
    if (control.dataset?.summaryLabel) return control.dataset.summaryLabel;
    const row = control.closest(".filter-row");
    if (row?.querySelector("label")?.textContent?.trim()) return row.querySelector("label").textContent.trim();
    const wrapperLabel = control.closest("label");
    if (wrapperLabel) {
      const clone = wrapperLabel.cloneNode(true);
      clone.querySelectorAll("input, select, textarea, option").forEach((node) => node.remove());
      const text = clone.textContent.trim();
      if (text) return text;
    }
    return control.name || control.id || "Selection";
  }

  function summaryValueFor(control) {
    if (control.type === "checkbox") return "Yes";
    if (control.tagName === "SELECT") return control.options?.[control.selectedIndex]?.text || control.value.trim();
    return control.value.trim();
  }

  function appendSummarySection(title, children) {
    const section = document.createElement("div");
    section.className = "summary-section";
    const heading = document.createElement("strong");
    heading.textContent = title;
    section.appendChild(heading);
    children.forEach((child) => section.appendChild(child));
    liveSummary.appendChild(section);
  }

  function emptySummaryText(text) {
    const empty = document.createElement("p");
    empty.className = "jp-text-muted";
    empty.textContent = text;
    return empty;
  }

  function makeSummaryList(items) {
    const list = document.createElement("ul");
    list.className = "summary-list";
    items.forEach(({ label: itemLabel, value }) => {
      const item = document.createElement("li");
      const label = document.createElement("strong");
      label.textContent = `${itemLabel}: `;
      item.appendChild(label);
      item.append(document.createTextNode(value));
      list.appendChild(item);
    });
    return list;
  }

  function compactListText(controlName, limit = 6) {
    const values = listValue(controlName);
    if (values.length <= limit) return values.join(", ");
    return `${values.slice(0, limit).join(", ")} +${values.length - limit} more`;
  }

  function isAdvancedControl(control) {
    return Boolean(control.closest(".advanced-detail-panel"));
  }

  function isSummaryControlVisible(control) {
    const closedDetails = control.closest("details:not([open])");
    return Boolean(
      control &&
      control.type !== "hidden" &&
      !control.classList.contains("hidden") &&
      !control.closest(".hidden") &&
      !control.closest(".canonical-hidden-fields") &&
      !closedDetails
    );
  }

  function isSummaryEmpty(control) {
    if (control === locationPreset) {
      return !control.value || (control.value === "custom" && !controlValue(locationCustom));
    }
    if (control === locationCustom) {
      return !isLocationCustomVisible() || !controlValue(control);
    }
    return isEmptyValue(control);
  }

  function markAdvancedTouched(control) {
    if (isAdvancedControl(control)) {
      control.dataset.userTouched = "true";
    }
  }

  function updateLiveSummary() {
    if (!form || !liveSummary) return;
    const controls = Array.from(form.querySelectorAll("[data-summary-field]"))
      .filter(isSummaryControlVisible);
    liveSummary.textContent = "";
    const active = controls.filter((control) => !isSummaryEmpty(control));

    const selectedItems = active.map((control) => ({
      label: isAdvancedControl(control) ? `Advanced - ${labelFor(control)}` : labelFor(control),
      value: summaryValueFor(control)
    }));
    if (selectedItems.length > 18) {
      selectedItems.splice(18, selectedItems.length - 18, {
        label: "More selections",
        value: `${selectedItems.length - 18} additional selections active`
      });
    }
    appendSummarySection(
      "Selected filters",
      selectedItems.length ? [makeSummaryList(selectedItems)] : [emptySummaryText("No filters applied.")]
    );

    const profileItems = [];
    const roleText = compactListText("manual_target_roles", 4);
    const skillText = compactListText("manual_skills", 8);
    const visaText = controlValue(namedControl("manual_visa_sponsorship"));
    if (roleText) profileItems.push({ label: "Target roles", value: roleText });
    if (skillText) profileItems.push({ label: "Skills", value: skillText });
    if (visaText) profileItems.push({ label: "Work authorization", value: visaText });
    appendSummarySection(
      "Parsed profile",
      profileItems.length ? [makeSummaryList(profileItems)] : [emptySummaryText("No parsed profile yet.")]
    );
  }

  function applySmartControl(control) {
    const value = control.value;

    if (control.id === "role_family_mode") {
      const modes = {
        "": ["", "", false],
        machine_learning: ["ml_related, research_ai", "ml_related, research_ai", true],
        ml_platform: ["ml_infra, ml_related", "ml_infra, ml_related", true],
        analytics_bi: ["", "analytics_entry, bi_analytics", false],
        data_pipelines: ["data_engineering", "data_engineering", true]
      };
      const [required, preferred, strict] = modes[value] || modes[""];
      setNamedValue("manual_required_role_families", required);
      setNamedValue("manual_preferred_role_families", preferred);
      setNamedValue("manual_strict_role_family", strict);
    }

    if (control.id === "seniority_mode") {
      const modes = {
        "": ["", "", "", ""],
        entry_2: ["2", "senior, staff_principal, lead_manager", "staff, principal, director, lead, manager", "senior, sr, iii"],
        early_4: ["4", "senior, staff_principal, lead_manager", "staff, principal, director, lead, manager", "senior, sr, iii"],
        no_staff_lead: ["", "staff_principal, lead_manager", "staff, principal, director, lead, manager", ""]
      };
      const [maxYears, excluded, hardReject, penalize] = modes[value] || modes[""];
      setNamedValue("manual_max_years_required", maxYears);
      setNamedValue("manual_excluded_seniority", excluded);
      setNamedValue("manual_hard_reject_seniority_terms", hardReject);
      setNamedValue("manual_penalize_seniority_terms", penalize);
    }

    if (control.id === "location_mode") {
      setNamedValue("manual_us_only", value === "us_only");
      setNamedValue("manual_strict_location", value === "us_only" || value === "strict_listed");
      if (value === "us_only" && !listValue("manual_location_preferences").some((item) => /^(us|u\.s\.|united states)$/i.test(item))) {
        appendListItems("manual_location_preferences", ["United States"]);
      }
      syncLocationControlsFromCanonical();
    }

    if (control.id === "sponsorship_mode") {
      setNamedValue("manual_needs_sponsorship", value === "needs");
      setNamedValue("manual_visa_sponsorship", value === "needs" ? "Needs H-1B / visa sponsorship." : "");
      if (value === "needs") {
        appendListItems("manual_dealbreakers", ["no sponsorship"]);
      } else {
        removeListItems("manual_dealbreakers", ["no sponsorship"]);
      }
    }

    if (control.id === "employment_mode") {
      if (value === "full_time_only") {
        setNamedValue("manual_excluded_employment_types", "contract, temporary, unpaid");
        appendListItems("manual_dealbreakers", ["contract", "temporary", "unpaid"]);
      } else {
        setNamedValue("manual_excluded_employment_types", "");
        removeListItems("manual_dealbreakers", ["contract", "temporary", "unpaid"]);
      }
    }

    if (
      control.id === "company_size_mode"
    ) {
      applyCompanyControls();
    }

    if (control.id === "salary_mode") {
      if (value === "ignore") {
        setNamedValue("manual_salary_min", "");
        setNamedValue("manual_salary_is_dealbreaker", false);
      } else {
        setNamedValue("manual_salary_is_dealbreaker", value === "hard_min");
      }
    }

    updateLiveSummary();
  }

  function setSmartControl(id, value) {
    const control = document.getElementById(id);
    if (control) control.value = value;
  }

  function setCompanySize(value) {
    if (companyControls.size) companyControls.size.value = value;
  }

  function syncCompanyControlsFromCanonical() {
    const preferredCompany = listValue("manual_preferred_company_types").map((item) => item.toLowerCase());
    if (preferredCompany.includes("large_company")) {
      setCompanySize("large");
    } else if (preferredCompany.includes("medium_company")) {
      setCompanySize("medium");
    } else if (preferredCompany.includes("small_company")) {
      setCompanySize("small");
    } else {
      setCompanySize("");
    }
    applyCompanyControls("inferred");
  }

  function applyCompanyControls(source = "user") {
    const preferred = [];
    const sizeMap = {
      large: "large_company",
      medium: "medium_company",
      small: "small_company"
    };
    const selectedSize = companyControls.size?.value || "";
    if (sizeMap[selectedSize]) preferred.push(sizeMap[selectedSize]);
    setListPreserving("manual_preferred_company_types", ["large_company", "medium_company", "small_company", "research_lab", "startup"], preferred, source);
    setListPreserving("manual_excluded_company_types", ["startup", "defense_military"], [], source);
    setNamedValue("manual_avoid_defense_or_clearance", false, source);
    updateLiveSummary();
  }

  function syncSmartControlsFromCanonical() {
    const requiredFamilies = listValue("manual_required_role_families").map((item) => item.toLowerCase());
    const preferredFamilies = listValue("manual_preferred_role_families").map((item) => item.toLowerCase());
    const strictRole = Boolean(controlValue(namedControl("manual_strict_role_family")));
    if (strictRole && requiredFamilies.includes("ml_infra")) {
      setSmartControl("role_family_mode", "ml_platform");
    } else if (strictRole && requiredFamilies.includes("data_engineering")) {
      setSmartControl("role_family_mode", "data_pipelines");
    } else if (strictRole && requiredFamilies.includes("ml_related")) {
      setSmartControl("role_family_mode", "machine_learning");
    } else if (preferredFamilies.includes("analytics_entry") || preferredFamilies.includes("bi_analytics")) {
      setSmartControl("role_family_mode", "analytics_bi");
    } else {
      setSmartControl("role_family_mode", "");
    }

    const maxYears = namedControl("manual_max_years_required")?.value || "";
    const excludedSeniority = listValue("manual_excluded_seniority").map((item) => item.toLowerCase());
    if (maxYears === "2" && excludedSeniority.includes("senior")) {
      setSmartControl("seniority_mode", "entry_2");
    } else if (maxYears === "4" && excludedSeniority.includes("senior")) {
      setSmartControl("seniority_mode", "early_4");
    } else if (excludedSeniority.includes("staff_principal") || excludedSeniority.includes("lead_manager")) {
      setSmartControl("seniority_mode", "no_staff_lead");
    } else {
      setSmartControl("seniority_mode", "");
    }

    if (controlValue(namedControl("manual_us_only"))) {
      setSmartControl("location_mode", "us_only");
    } else if (controlValue(namedControl("manual_strict_location"))) {
      setSmartControl("location_mode", "strict_listed");
    } else {
      setSmartControl("location_mode", "");
    }

    setSmartControl("sponsorship_mode", controlValue(namedControl("manual_needs_sponsorship")) ? "needs" : "");

    const excludedEmployment = listValue("manual_excluded_employment_types").map((item) => item.toLowerCase());
    setSmartControl(
      "employment_mode",
      ["contract", "temporary", "unpaid"].every((item) => excludedEmployment.includes(item)) ? "full_time_only" : ""
    );

    syncCompanyControlsFromCanonical();

    if (!namedControl("manual_salary_min")?.value) {
      setSmartControl("salary_mode", "");
    } else {
      setSmartControl("salary_mode", controlValue(namedControl("manual_salary_is_dealbreaker")) ? "hard_min" : "");
    }
  }

  function populateFields(formFields, fieldSources) {
    Object.entries(formFields || {}).forEach(([name, value]) => {
      const control = form?.elements?.[name];
      const isRadioNodeList = typeof RadioNodeList !== "undefined" && control instanceof RadioNodeList;
      if (!control || isRadioNodeList) return;
      setControlValue(control, value);
      const source = fieldSources?.[name] || (isEmptyValue(control) ? "empty" : "inferred");
      updateBadge(control, source === "empty" ? "empty" : "inferred");
    });
    syncLocationControlsFromCanonical();
    syncSmartControlsFromCanonical();
    markFiltersReady();
    updateLiveSummary();
  }

  function hasReadableProfileInput() {
    return Boolean((profileText?.value || "").trim() || resumeFile?.files?.length);
  }

  function prepareReadableProfileInput() {
    if (hasReadableProfileInput()) return true;

    const selectedPreset = personaPreset?.value || "";
    if (selectedPreset && selectedPreset !== "manual" && profileText && demoTexts[selectedPreset]) {
      profileText.value = demoTexts[selectedPreset];
      clearFiltersReady();
      return true;
    }

    setMessage(parseError, "Add profile/resume text or choose a demo persona before reading the profile.");
    setMessage(parseNotes, []);
    profileText?.focus();
    return false;
  }

  async function readProfile(options = {}) {
    if (!prepareReadableProfileInput()) return;

    const selectedPreset = personaPreset?.value || "";
    const keepPersona = Boolean(options.keepPersona || (selectedPreset && selectedPreset !== "manual"));
    const body = new FormData();
    body.append("profile_text", profileText?.value || "");
    body.append("persona", selectedPreset || "manual");
    body.append("parser_mode", parserMode?.value || "rule_fallback");
    body.append("parser_provider", parserProvider?.value || "gemini");
    if (parserModel?.value) {
      body.append("parser_model", parserModel.value);
    }
    if (parserApiKey?.value) {
      body.append("parser_api_key", parserApiKey.value);
    }
    if (resumeFile?.files?.length) {
      body.append("resume_pdf", resumeFile.files[0]);
    }

    setParseBusy(true);
    setMessage(parseError, []);
    setMessage(parseNotes, []);
    rememberParserApiSettings();

    try {
      const response = await fetch("/parse-profile", {
        method: "POST",
        body
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const payload = await response.json();
      populateFields(payload.form_fields || {}, payload.field_sources || {});
      const methodNote = payload.parse_method ? [`Parse method: ${payload.parse_method}`] : [];
      setMessage(parseNotes, [...methodNote, ...(payload.notes || [])]);
      if (parserMode?.value === "llm" && String(payload.parse_method || "").startsWith("llm_failed")) {
        setMessage(
          parseError,
          "Gemini API did not complete; local fallback was used. Check the API key, model access, and network."
        );
      }
      if (!keepPersona && personaPreset) personaPreset.value = "manual";
    } catch (error) {
      setMessage(parseError, `Profile parsing failed. ${error.message}`);
    } finally {
      setParseBusy(false);
    }
  }

  parseButton?.addEventListener("click", (event) => {
    event?.preventDefault();
    event?.stopPropagation();
    setMatchBusy(false);
    readProfile();
  });

  parserMode?.addEventListener("change", updateParseButtonText);

  form?.addEventListener("submit", (event) => {
    if (parseInProgress) {
      event.preventDefault();
      setMatchBusy(false);
      setMessage(parseError, "Wait for the profile reader to finish before running matching.");
      return;
    }
    setMatchBusy(true);
  });

  fieldsContainer?.addEventListener("input", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement) && !(target instanceof HTMLTextAreaElement) && !(target instanceof HTMLSelectElement)) return;
    markFiltersReady();
    markAdvancedTouched(target);
    if (target === locationPreset || target === locationCustom) {
      if (target === locationPreset) {
        locationCustom.dataset.expanded = target.value === "custom" ? "true" : "";
      }
      updateLocationCanonical("user");
      return;
    }
    if (target.matches("[data-smart-control]")) {
      applySmartControl(target);
      return;
    }
    updateBadge(target, isEmptyValue(target) ? "empty" : "user");
    updateLiveSummary();
  });

  fieldsContainer?.addEventListener("change", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement) && !(target instanceof HTMLTextAreaElement) && !(target instanceof HTMLSelectElement)) return;
    markFiltersReady();
    markAdvancedTouched(target);
    if (target === locationPreset || target === locationCustom) {
      if (target === locationPreset) {
        locationCustom.dataset.expanded = target.value === "custom" ? "true" : "";
      }
      updateLocationCanonical("user");
      return;
    }
    if (target.matches("[data-smart-control]")) {
      applySmartControl(target);
      return;
    }
    updateBadge(target, isEmptyValue(target) ? "empty" : "user");
    updateLiveSummary();
  });

  locationAddButton?.addEventListener("click", () => {
    if (!locationCustom) return;
    markFiltersReady();
    locationCustom.dataset.expanded = "true";
    if (locationPreset && !locationPreset.value) {
      locationPreset.value = "custom";
    }
    refreshLocationCustomVisibility();
    updateLocationCanonical("user");
    locationCustom.focus();
  });

  document.querySelectorAll("[data-demo]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.getAttribute("data-demo");
      if (!key || !profileText) return;
      profileText.value = demoTexts[key] || "";
      clearFiltersReady();
      if (personaPreset) personaPreset.value = "manual";
      profileText.focus();
      readProfile();
    });
  });

  personaPreset?.addEventListener("change", () => {
    const key = personaPreset.value;
    if (!key || key === "manual" || !profileText) {
      clearFiltersReady();
      updateLiveSummary();
      return;
    }
    profileText.value = demoTexts[key] || "";
    clearFiltersReady();
    readProfile({ keepPersona: true });
  });

  profileText?.addEventListener("input", clearFiltersReady);
  resumeFile?.addEventListener("change", clearFiltersReady);
  window.addEventListener("pageshow", () => {
    setParseBusy(false);
    setMatchBusy(false);
    refreshLocationCustomVisibility();
    updateLiveSummary();
  });

  refreshLocationCustomVisibility();
  updateParseButtonText();
  updateLiveSummary();
  window.setTimeout(() => {
    refreshLocationCustomVisibility();
    updateParseButtonText();
    updateLiveSummary();
  }, 0);
});
