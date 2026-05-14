(function(){
  var p = document.querySelector(".prose");
  if (!p) return;
  var hr = p.querySelector("hr");
  if (!hr) return;
  var h2n = 0, h3n = 0;
  var el = hr.nextElementSibling;
  while (el) {
    var tag = el.tagName;
    if (tag === "H2") {
      h2n++;
      h3n = 0;
      el.setAttribute("data-num", h2n + ". ");
    } else if (tag === "H3") {
      h3n++;
      el.setAttribute("data-num", h2n + "." + h3n + " ");
    }
    el = el.nextElementSibling;
  }
})();
