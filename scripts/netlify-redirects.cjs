/**
 * Netlify: proxy /api/* to your FastAPI host so the browser uses same-origin fetch (no CORS).
 * Set env BACKEND_URL in Netlify UI, e.g. https://your-app.up.railway.app (no trailing slash).
 */
const fs = require("fs");
const path = require("path");
const pub = path.join(__dirname, "..", "web", "public");
const backend = (process.env.BACKEND_URL || "").trim().replace(/\/$/, "");
const lines = [];
if (backend) {
  lines.push(`/api/*\t${backend}/api/:splat\t200`);
  console.log("Netlify _redirects: proxy /api/* ->", backend);
} else {
  console.warn(
    "BACKEND_URL is not set. Set it in Netlify: Site settings → Environment variables → BACKEND_URL\n" +
      "Example: https://your-api.onrender.com\n" +
      "Without it, /api/* will fall through and photo screening will fail."
  );
}
lines.push("/*\t/index.html\t200");
fs.writeFileSync(path.join(pub, "_redirects"), lines.join("\n") + "\n", "utf8");
console.log("Wrote", path.join(pub, "_redirects"));