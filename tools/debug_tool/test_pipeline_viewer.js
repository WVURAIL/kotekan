// Run with: node tools/debug_tool/test_pipeline_viewer.js
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const elements = new Map();
const context = vm.createContext({
    navigator: { appName: "Netscape", userAgent: "Trident/7.0; rv:11.0" },
    document: { getElementById: id => elements.get(id) },
});
vm.runInContext(fs.readFileSync(path.join(__dirname, "static/pipeline_viewer.js"), "utf8"), context);
assert.equal(context.isIE(), true);
context.navigator.userAgent = "Chrome/120.0";
assert.equal(context.isIE(), false);

for (const id of ["stage_button", "stage/tracker", "stage/tracker_time",
    "stage/tracker_cur", "stage/tracker_min", "stage/tracker_max",
    "stage/tracker_avg", "stage/tracker_std"]) {
    const element = {};
    Object.defineProperty(element, "innerHTML", {
        set() { throw new Error("Untrusted values must not be parsed as HTML"); },
    });
    elements.set(id, element);
}
const payload = "<img src=x onerror=alert(1)>";
context.update_table("stage", "tracker", [payload, 1, 2, 3, 4, 0, "units"], false);
assert.equal(elements.get("stage/tracker_cur").textContent, payload);
console.log("Viewer browser detection and text rendering checks passed.");
