document.querySelectorAll('.tabs button').forEach(function (b) {
  b.addEventListener('click', function () {
    document.querySelectorAll('.tabs button').forEach(function (o) {
      o.setAttribute('aria-selected', String(o === b));
      document.getElementById('lb-' + o.dataset.tab).hidden = o !== b;
    });
  });
});
