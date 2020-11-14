open_tag = """
<!DOCTYPE html>
<html>
<head>
<title>Plagiarism report</title>
</head>
<style>

body {
    font-family: Helvetica, sans-serif;
    background: #222;
    margin: 0;
    color: lightgray;
    height:100%
}

.module-border-wrap {
    padding: 1rem;
    background: linear-gradient(to right, red, purple);
    padding: 2px;
}

.module {
  background: #222;
  padding: 2rem;
}

.tooltip {
    position: relative;
    display: inline-block;
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

.paragraph {
    border-radius: 10px;
    text-indent: 40px;
    padding: 3px;
}

hr {
    border: none;
    color: purple;
    background-color: purple;
    height: 2px;
    width: 400px;
}

</style>
<body>
    <div class="module-border-wrap">
        <div class="module">
            <h1 style="text-align:center">Plagiarism report</h1>
            <hr>
"""

parahraph_tag = """
<div class="tooltip ">
        <p class="paragraph" style = "background:rgba({0}, .4); border: 2px solid rgba({0}, 1); ">{1}</p>
        <span class="tooltiptext">{2}</span>
</div>
<br>
"""

end_report = """
<hr>
<div>
    <h3 style = "text-indent: 40px;">Document originality average: {0}%</h3>
</div>
"""

close_tag = """
        </div>
    </div>
</body>
</html> """


def generate_html(paragraphs):
    html_content = ""
    html_content += open_tag
    plagiat_type = {0: "Non plagiat", 1: "Heavily paraphrased",
                    2: "Lightly paraphrased", 3: "Copypasted"}
    all_percents = []
    for paragraph in paragraphs:
        percents = int(paragraph['prob'][0] * 100)
        all_percents.append(percents)
        percent_text = "Original percent: {0}%<br>Task type: {1}<br>Plagiat type: {2}".format(
            percents, paragraph['task'], plagiat_type[paragraph['class']])
        color = '%02d,%02d,%02d' % (
            int((1 - paragraph['prob'][0]) * 256), int(paragraph['prob'][0] * 256), 0)
        html_content += parahraph_tag.format(
            color, paragraph['text'], percent_text)
    html_content += end_report.format(int(sum(all_percents) / len(all_percents)))
    html_content += close_tag

    f = open("plagiat_report.html", "w")
    f.write(html_content)
    f.close()
