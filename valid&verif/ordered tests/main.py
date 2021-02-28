from formatter import formatter
from formatter import analyzer
if __name__ == '__main__':
	indent_list = list()
	long_line_list = list()
	with open("config.txt") as config:
		for line in config:
			indent, long_line = line.split(" ")
			indent_list.append(indent)
			long_line_list.append(long_line)
	for i, indent in enumerate(indent_list):
		print(str(i) + " --------------------")
		print("indent_const: " + indent)
		print("long_line_const: " + long_line_list[i])
	print("Enter number of fromatting: ")
	num = int(input())	
	fmt = formatter.Formatter("test.kt", indent_list[num], long_line_list[num])
	fmt.format_file()

	print(fmt.print_finished_content())
	#analyzer = analyzer.Analyzer("test.kt")
	#analyzer.analyze()