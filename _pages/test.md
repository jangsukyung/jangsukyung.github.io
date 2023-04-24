---
title: "Test"
layout: archive
permalink: /Test
---


{% assign posts = site.categories.Test %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}