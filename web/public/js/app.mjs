const API = "";

function httpErrorMessage(data, fallback) {
  const d = data?.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((x) => x?.msg || JSON.stringify(x)).join("; ");
  return fallback;
}

const VIEWS = [
  "home",
  "signup",
  "login",
  "dashboard",
  "child-form",
  "photo",
  "photo-stop",
  "questionnaire",
  "results",
];

const QCHAT = [
  { n: 1, t: "Does your child look at you when you call his/her name?" },
  { n: 2, t: "How easy is it for you to get eye contact with your child?" },
  { n: 3, t: "Does your child point to indicate that s/he wants something?" },
  { n: 4, t: "Does your child point to share interest with you?" },
  { n: 5, t: "Does your child pretend? (e.g. care for dolls, talk on toy phone)" },
  { n: 6, t: "Does your child follow where you're looking?" },
  { n: 7, t: "Does your child comfort upset family members?" },
  { n: 8, t: "Would you describe your child's first words as typical?" },
  { n: 9, t: "Does your child use simple gestures? (e.g. wave goodbye)" },
  { n: 10, t: "Does your child stare at nothing with no apparent purpose?" },
];

const MCHAT = [
  { n: 11, t: "If you point at something across the room, does your child look at it?" },
  { n: 12, t: "Have you ever wondered if your child might be deaf?" },
  { n: 13, t: "Does your child like climbing on things?" },
  { n: 14, t: "Does your child make unusual finger movements near his/her eyes?" },
  { n: 15, t: "Is your child interested in other children?" },
  { n: 16, t: "Does your child show you things by bringing them to you?" },
  { n: 17, t: "When you smile at your child, does he/she smile back at you?" },
  { n: 18, t: "Does your child get upset by everyday noises?" },
  { n: 19, t: "Does your child walk?" },
  { n: 20, t: "Does your child try to copy what you do?" },
  { n: 21, t: "Does your child try to get you to watch him/her?" },
  { n: 22, t: "Does your child understand when you tell him/her to do something?" },
  { n: 23, t: "Does your child look at your face to see how you feel about something new?" },
  { n: 24, t: "Does your child like movement activities?" },
];

let firebaseApp = null;
let firebaseAuth = null;
let firebaseDb = null;

let activeUser = null;
let activeChild = null; // {id, ...fields}
let editingChildId = null;

let lastFaceResult = null;
let lastInferenceResult = null;
let lastReportText = "";

function getStoredGender(record) {
  return String(record?.gender || record?.sex || "");
}

function getGenderField() {
  return document.getElementById("ch-gender") || document.getElementById("ch-sex");
}

function toast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("show");
  window.clearTimeout(toast._t);
  toast._t = window.setTimeout(() => el.classList.remove("show"), 2600);
}

function showView(name) {
  VIEWS.forEach((v) => {
    const el = document.getElementById(`view-${v}`);
    if (!el) return;
    el.classList.toggle("active", v === name);
  });
}

function emailFromUsername(username) {
  const slug = String(username || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "");
  if (!slug) throw new Error("Username is required.");
  return `${slug}@parents-aid.local`;
}

async function initFirebase() {
  const cfg = window.FIREBASE_CONFIG;
  if (!cfg?.apiKey || String(cfg.apiKey).includes("YOUR")) {
    toast("Configure Firebase in web/public/firebase-config.js");
    return false;
  }

  const { initializeApp } = await import("https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js");
  const { getAuth, onAuthStateChanged } = await import("https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js");
  const { getFirestore } = await import("https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js");

  firebaseApp = initializeApp(cfg);
  firebaseAuth = getAuth(firebaseApp);
  firebaseDb = getFirestore(firebaseApp);

  onAuthStateChanged(firebaseAuth, (user) => {
    activeUser = user;
    const chip = document.getElementById("authChip");
    chip.textContent = user ? `Signed in` : "";
    if (user) {
      showView("dashboard");
      loadChildren();
    } else {
      showView("home");
    }
  });

  return true;
}

async function loadChildren() {
  if (!activeUser || !firebaseDb) return;
  const { collection, getDocs } = await import(
    "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js"
  );
  const col = collection(firebaseDb, "users", activeUser.uid, "children");
  const snap = await getDocs(col);
  const root = document.getElementById("childrenList");
  root.innerHTML = "";
  if (snap.empty) {
    root.innerHTML = `<div class="muted">No child profiles yet.</div>`;
    return;
  }
  snap.forEach((docSnap) => {
    const d = docSnap.data();
    const row = document.createElement("div");
    row.className = "child-item";
    row.innerHTML = `
      <div>
        <div style="font-weight:800">${escapeHtml(d.childName || "Child")}</div>
        <div class="small">${escapeHtml(String(d.age_mons || ""))} months · ${escapeHtml(getStoredGender(d))}</div>
      </div>
      <div class="row">
        <button class="btn secondary" data-action="edit" data-id="${docSnap.id}">Edit</button>
        <button class="btn" data-action="start" data-id="${docSnap.id}">Start screening</button>
      </div>
    `;
    row.querySelector('[data-action="edit"]').addEventListener("click", () => openChildForm(docSnap.id, d));
    row.querySelector('[data-action="start"]').addEventListener("click", () => startScreeningForChild(docSnap.id, d));
    root.appendChild(row);
  });
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function openChildForm(childId = null, data = null) {
  editingChildId = childId;
  document.getElementById("childFormTitle").textContent = childId ? "Edit child profile" : "Add child profile";
  document.getElementById("ch-name").value = data?.childName || "";
  document.getElementById("ch-age").value = data?.age_mons ?? "";
  const genderField = getGenderField();
  if (genderField) genderField.value = getStoredGender(data);
  document.getElementById("ch-jaundice").value = data?.jaundice || "";
  document.getElementById("ch-fam").value = data?.family_mem_with_asd || "";
  showView("child-form");
}

async function saveChildProfile() {
  if (!activeUser || !firebaseDb) return toast("Not signed in");
  const childName = document.getElementById("ch-name").value.trim();
  const age_mons = Number(document.getElementById("ch-age").value);
  const gender = getGenderField()?.value || "";
  const jaundice = document.getElementById("ch-jaundice").value;
  const family = document.getElementById("ch-fam").value;
  if (!childName) return toast("Child name is required");
  if (!Number.isFinite(age_mons) || age_mons < 12 || age_mons > 60) return toast("Age must be 12–60 months");
  if (!gender || !jaundice || !family) return toast("Please complete all fields");

  const { addDoc, collection, doc, serverTimestamp, setDoc } = await import(
    "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js"
  );

  const payload = { childName, age_mons, gender, jaundice, family_mem_with_asd: family, updatedAt: serverTimestamp() };
  if (!editingChildId) {
    await addDoc(collection(firebaseDb, "users", activeUser.uid, "children"), {
      ...payload,
      createdAt: serverTimestamp(),
    });
    toast("Child profile saved");
  } else {
    await setDoc(doc(firebaseDb, "users", activeUser.uid, "children", editingChildId), payload, { merge: true });
    toast("Child profile updated");
  }
  editingChildId = null;
  showView("dashboard");
  await loadChildren();
}

function startScreeningForChild(childId, data) {
  activeChild = { id: childId, ...data, gender: getStoredGender(data) };
  lastFaceResult = null;
  lastInferenceResult = null;
  lastReportText = "";
  document.getElementById("photoFile").value = "";
  showView("photo");
}

async function runPhoto() {
  const f = document.getElementById("photoFile").files?.[0];
  if (!f) return toast("Choose a photo first");
  try {
    const fd = new FormData();
    fd.append("file", f, f.name);
    const res = await fetch(`${API}/api/face/predict`, { method: "POST", body: fd });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(httpErrorMessage(data, `HTTP ${res.status}`));
    lastFaceResult = data;

    if (!data.is_autistic) {
      document.getElementById("photoStopText").textContent =
        "Based on the photo screening step, this workflow does not continue to the behavioural questionnaires. " +
        "If you still have concerns, please speak with a qualified professional. You can retake the photo step.";
      showView("photo-stop");
      return;
    }

    renderQuestionnaire();
    showView("questionnaire");
  } catch (e) {
    toast(String(e.message || e));
  }
}

function renderQuestionnaire() {
  const root = document.getElementById("qRoot");
  root.innerHTML = "";

  const h = document.createElement("div");
  h.innerHTML = `<div class="badge ok" style="margin-bottom:10px">Q-CHAT-10</div>`;
  root.appendChild(h);

  for (const q of QCHAT) {
    const wrap = document.createElement("div");
    wrap.className = "qblock";
    wrap.innerHTML = `
      <div class="qtitle">${q.n}. ${escapeHtml(q.t)}</div>
      <select class="form-control" id="qchat_${q.n}" style="width:100%;height:40px;border-radius:10px;border:2px solid #cde4e4">
        <option value="">Select…</option>
        <option>A</option><option>B</option><option>C</option><option>D</option><option>E</option>
      </select>
    `;
    root.appendChild(wrap);
  }

  const h2 = document.createElement("div");
  h2.innerHTML = `<div class="badge ok" style="margin:16px 0 10px">M-CHAT-R</div>`;
  root.appendChild(h2);

  for (const q of MCHAT) {
    const wrap = document.createElement("div");
    wrap.className = "qblock";
    wrap.innerHTML = `
      <div class="qtitle">${q.n}. ${escapeHtml(q.t)}</div>
      <select class="form-control" id="mchat_${q.n}" style="width:100%;height:40px;border-radius:10px;border:2px solid #cde4e4">
        <option value="">Select…</option>
        <option>Yes</option>
        <option>No</option>
      </select>
    `;
    root.appendChild(wrap);
  }
}

function collectQuestionnairePayload() {
  const qchat_answers = {};
  for (const q of QCHAT) {
    const v = document.getElementById(`qchat_${q.n}`).value;
    if (!v) throw new Error(`Please answer Q-CHAT question ${q.n}`);
    qchat_answers[String(q.n)] = v;
  }
  const mchat_answers = {};
  for (const q of MCHAT) {
    const v = document.getElementById(`mchat_${q.n}`).value;
    if (!v) throw new Error(`Please answer M-CHAT question ${q.n}`);
    mchat_answers[String(q.n)] = v;
  }
  return { qchat_answers, mchat_answers };
}

async function submitQuestionnaire() {
  if (!activeChild) return toast("No active child profile");
  try {
    const { qchat_answers, mchat_answers } = collectQuestionnairePayload();
    const body = {
      age_mons: Number(activeChild.age_mons),
      gender: getStoredGender(activeChild),
      jaundice: String(activeChild.jaundice),
      family_mem_with_asd: String(activeChild.family_mem_with_asd),
      qchat_answers,
      mchat_answers,
    };
    const res = await fetch(`${API}/api/screen/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(httpErrorMessage(data, `HTTP ${res.status}`));
    lastInferenceResult = data;
    lastReportText = "";
    await maybeSaveScreeningToFirestore({ face: lastFaceResult, inference: data });
    renderResults(data);
    showView("results");
  } catch (e) {
    toast(String(e.message || e));
  }
}

async function maybeSaveScreeningToFirestore(payload) {
  try {
    if (!activeUser || !firebaseDb || !activeChild?.id) return;
    const { addDoc, collection, serverTimestamp } = await import(
      "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js"
    );
    await addDoc(collection(firebaseDb, "users", activeUser.uid, "children", activeChild.id, "screenings"), {
      createdAt: serverTimestamp(),
      ...payload,
    });
  } catch {
    // non-fatal
  }
}

function renderResults(result) {
  const root = document.getElementById("resultsRoot");
  const probs = result.class_probabilities || {};
  const probRows = Object.entries(probs)
    .map(([k, v]) => `<div><strong>${escapeHtml(k)}</strong>: ${(Number(v) * 100).toFixed(1)}%</div>`)
    .join("");
  root.innerHTML = `
    <div style="display:grid;gap:10px">
      <div><span class="badge ok">Score</span> <strong>${result.screening_score}/${result.screening_score_max}</strong></div>
      <div><span class="badge ok">Score risk</span> <strong>${escapeHtml(result.score_risk_level)}</strong></div>
      <div><span class="badge ok">Referral</span> ${escapeHtml(result.referral_interpretation)}</div>
      <div><span class="badge ok">ML default</span> ${escapeHtml(result.prediction_default?.predicted_label || "")}</div>
      <div><span class="badge ok">ML screening</span> ${escapeHtml(result.prediction_screening?.predicted_label || "")}</div>
      <div style="margin-top:8px"><strong>Class probabilities</strong></div>
      ${probRows}
      <div class="small" style="margin-top:10px">${escapeHtml(result.disclaimer || "")}</div>
    </div>
  `;
  document.getElementById("reportBox").textContent = "";
  document.getElementById("btnDownloadPdf").disabled = true;
}

async function genReport() {
  if (!lastInferenceResult) return toast("No results yet");
  try {
    const res = await fetch(`${API}/api/report/llm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ inference_result: lastInferenceResult }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(httpErrorMessage(data, `HTTP ${res.status}`));
    lastReportText = data.report_text || "";
    document.getElementById("reportBox").textContent = lastReportText;
    document.getElementById("btnDownloadPdf").disabled = !lastReportText;
    toast("Report generated");
  } catch (e) {
    toast(String(e.message || e));
  }
}

async function downloadPdf() {
  if (!lastInferenceResult || !lastReportText) return toast("Generate the report first");
  try {
    const res = await fetch(`${API}/api/report/pdf`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ inference_result: lastInferenceResult, report_text: lastReportText }),
    });
    if (!res.ok) {
      const t = await res.text();
      throw new Error(t || `HTTP ${res.status}`);
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `aid-report-${Date.now()}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (e) {
    toast(String(e.message || e));
  }
}

async function signup() {
  if (!firebaseAuth || !firebaseDb) return toast("Firebase not configured");
  const { createUserWithEmailAndPassword } = await import("https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js");
  const { doc, serverTimestamp, setDoc } = await import("https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js");
  const username = document.getElementById("su-username").value.trim();
  const password = document.getElementById("su-pass").value;
  const email = emailFromUsername(username);
  const cred = await createUserWithEmailAndPassword(firebaseAuth, email, password);
  await setDoc(doc(firebaseDb, "users", cred.user.uid), {
    username,
    emailSynthetic: email,
    createdAt: serverTimestamp(),
  });
  toast("Account created");
}

async function login() {
  if (!firebaseAuth) return toast("Firebase not configured");
  const { signInWithEmailAndPassword } = await import("https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js");
  const username = document.getElementById("lg-username").value.trim();
  const password = document.getElementById("lg-pass").value;
  const email = emailFromUsername(username);
  await signInWithEmailAndPassword(firebaseAuth, email, password);
  toast("Welcome back");
}

async function logout() {
  if (!firebaseAuth) return;
  const { signOut } = await import("https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js");
  await signOut(firebaseAuth);
}

function wire() {
  document.getElementById("btnGoLogin").addEventListener("click", () => showView("login"));
  document.getElementById("btnGoSignup").addEventListener("click", () => showView("signup"));
  document.getElementById("btnSignupBack").addEventListener("click", () => showView("home"));
  document.getElementById("btnLoginBack").addEventListener("click", () => showView("home"));
  document.getElementById("btnSignup").addEventListener("click", () => signup().catch((e) => toast(String(e.message || e))));
  document.getElementById("btnLogin").addEventListener("click", () => login().catch((e) => toast(String(e.message || e))));

  document.getElementById("btnLogout").addEventListener("click", () => logout().catch((e) => toast(String(e.message || e))));
  document.getElementById("btnAddChild").addEventListener("click", () => openChildForm(null, null));
  document.getElementById("btnSaveChild").addEventListener("click", () => saveChildProfile().catch((e) => toast(String(e.message || e))));
  document.getElementById("btnChildCancel").addEventListener("click", () => {
    editingChildId = null;
    showView("dashboard");
  });

  document.getElementById("btnPhotoBack").addEventListener("click", () => showView("dashboard"));
  document.getElementById("btnRunPhoto").addEventListener("click", () => runPhoto());

  document.getElementById("btnRetakePhoto").addEventListener("click", () => showView("photo"));
  document.getElementById("btnStopToDash").addEventListener("click", () => showView("dashboard"));

  document.getElementById("btnQsBack").addEventListener("click", () => showView("photo"));
  document.getElementById("btnSubmitQs").addEventListener("click", () => submitQuestionnaire());

  document.getElementById("btnGenReport").addEventListener("click", () => genReport());
  document.getElementById("btnDownloadPdf").addEventListener("click", () => downloadPdf());
  document.getElementById("btnRestartPhoto").addEventListener("click", () => {
    lastInferenceResult = null;
    lastReportText = "";
    document.getElementById("photoFile").value = "";
    showView("photo");
  });
  document.getElementById("btnResultsDash").addEventListener("click", () => showView("dashboard"));
}

await initFirebase();
wire();
