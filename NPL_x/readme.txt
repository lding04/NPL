首先cd到项目根目录
使用python运行
cmd /C "set PYTHONPATH=. && python npl\crawl.py"
直接运行exe文件
dist\crawl.exe

输入文件默认为npl_data/input/template.csv可以使用选项-f指定输入文件
输出目录默认为npl_data/output/可以使用选项-o指定输出目录