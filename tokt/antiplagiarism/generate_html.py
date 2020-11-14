open_tag = """
<!DOCTYPE html>
<html>
<head>
<title>Plagiarizm report</title>
</head>
<style>
.tooltip {
  position: relative;
  display: inline-block;
  font-family: Helvetica, sans-serif;
}

.tooltip .tooltiptext {
  visibility: hidden;
  width: 210px;
  background-color: black;
  color: #fff;
  border-radius: 6px;
  padding: 5px 0;
  position: absolute;
  z-index: 1;
  top: 100%;
  left: 50%;
  margin-left: -60px;
}

.tooltip .tooltiptext::after {
  content: "";
  position: absolute;
  bottom: 100%;
  left: 50%;
  margin-left: -5px;
  border-width: 5px;
  border-style: solid;
  border-color: transparent transparent black transparent;
}

.tooltip:hover .tooltiptext {
  visibility: visible;
}
</style>
<body">
"""

parahraph_tag = """
<div class="tooltip">
    <p style = "background-color:{0}; text-indent: 40px;">{1}</p>
    <span class="tooltiptext">{2}</span>
</div>
<br>
"""

close_tag = """
</body>
</html> """


def generate_html(paragraphs):
    html_content = ""
    html_content += open_tag
    plagiat_type = {0: "Non pagiat", 1: "Heavily paraphrased",
                    2: "Lightly paraphrased", 3: "Copypasted"}
    for paragraph in paragraphs:
        percents = int(paragraph['prob'][0] * 100)
        percent_text = "Original percent: {0}%<br>Task type: {1}<br>Plagiat type: {2}".format(
            percents, paragraph['task'], plagiat_type[paragraph['class']])
        color = '#%02x%02x%02x' % (
            int((1 - paragraph['prob'][0]) * 256), int(paragraph['prob'][0] * 256), 0)
        html_content += parahraph_tag.format(
            color, paragraph['text'], percent_text)
    html_content += close_tag

    f = open("plagiat_report.html", "w")
    f.write(html_content)
    f.close()
