document.addEventListener("DOMContentLoaded", function(event) {
	document.querySelectorAll(".wy-menu.wy-menu-vertical > ul.current > li > a").forEach(a => a.addEventListener("click", e=>{
		f = document.querySelector(".wy-menu.wy-menu-vertical > ul.current > li > ul")
		if (f.style.display=='none') { f.style.display='block'; } else f.style.display = 'none'
	}));
	document.querySelectorAll(".headerlink").forEach(a => a.text="\u{1F517}");
});
