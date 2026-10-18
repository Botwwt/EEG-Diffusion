/* Progressive enhancement: all reading panels remain available without JS. */
"use strict";

const tabs = [...document.querySelectorAll('[role="tab"]')];
const panels = [...document.querySelectorAll('[role="tabpanel"]')];

function selectPaper(index, focus = false) {
  tabs.forEach((tab, position) => {
    const selected = position === index;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    panels[position].hidden = !selected;
  });
  if (focus) tabs[index].focus();
}

tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => selectPaper(index));
  tab.addEventListener("keydown", (event) => {
    let destination = index;
    if (event.key === "ArrowRight") destination = (index + 1) % tabs.length;
    else if (event.key === "ArrowLeft") destination = (index + tabs.length - 1) % tabs.length;
    else if (event.key === "Home") destination = 0;
    else if (event.key === "End") destination = tabs.length - 1;
    else return;
    event.preventDefault();
    selectPaper(destination, true);
  });
});
selectPaper(0);

const dialog = document.getElementById("image-dialog");
let opener;
function showImage(image) {
  opener = image;
  const large = document.getElementById("large-image");
  large.src = image.src;
  large.alt = image.alt;
  document.getElementById("large-caption").textContent = image.alt;
  dialog.showModal();
}
document.querySelectorAll(".zoomable").forEach((image) => {
  image.addEventListener("click", () => showImage(image));
  image.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      showImage(image);
    }
  });
});
document.getElementById("close-image").addEventListener("click", () => dialog.close());
dialog.addEventListener("close", () => opener?.focus());
dialog.addEventListener("click", (event) => {
  const rectangle = dialog.getBoundingClientRect();
  if (event.target === dialog && (event.clientX < rectangle.left || event.clientX > rectangle.right ||
      event.clientY < rectangle.top || event.clientY > rectangle.bottom)) dialog.close();
});

document.getElementById("copy-code").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  const text = document.getElementById("quickstart").textContent;
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = "已复制";
    document.getElementById("copy-status").textContent = "运行命令已复制";
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.getElementById("quickstart"));
    selection.removeAllRanges();
    selection.addRange(range);
    document.getElementById("copy-status").textContent = "请复制已选中的命令";
  }
});
document.getElementById("print-report").addEventListener("click", () => window.print());

const links = [...document.querySelectorAll(".contents a")];
const observer = new IntersectionObserver((entries) => {
  const visible = entries.filter((entry) => entry.isIntersecting);
  if (!visible.length) return;
  const id = visible[0].target.id;
  links.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === `#${id}`));
}, { rootMargin: "-10% 0px -65% 0px" });
document.querySelectorAll(".report-section").forEach((section) => observer.observe(section));
