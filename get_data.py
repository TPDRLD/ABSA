text = []
txt = open('sample_reviews_xml.xml', 'r')
for line in txt:
	if '<text>' in line:
		line = line.replace('<text>', '')
		line = line.replace('</text>','')
		line=line.strip()
		text.append(line)

print text