import re

keys = [
    'sarhan2020evaluation', 'yan2023deep', 'lundberg2017unified', 'ribeiro2016why',
    'hnamte2024iot', 'pecorella2021iot', 'moustafa2015bot', 'iomt2024dataset',
    'ciciot2025dataset', 'roopak2019deep', 'verma2020machine', 'kumar2021eddos',
    'latif2020lightweight', 'sgouras2022detecting', 'zhao2020hybrid', 'almimi2023xai',
    'hosseini2024attention', 'susilo2020security', 'jiang2020network', 'yin2020deep',
    'zhou2021ddos', 'shaaban2023biomt', 'basnet2022deep', 'dong2021novel',
    'liu2022explainable', 'gad2021hybrid', 'bhamare2020machine', 'churcher2021machine',
    'soe2020machine', 'abbasi2021deep', 'gu2022lightweight', 'wu2023lightweight',
    'alomari2021hybrid', 'habib2023multi'
]

names = [
    'Yan', 'Han', 'Hnamte', 'Hussain', 'Moustafa', 'Slay', 'Gharib', 'Lashkari',
    'Verma', 'Ranga', 'Kumar', 'Latif', 'Sgouras', 'Zhao', 'Al-Mimi', 'Mimi',
    'Hosseini', 'Miri', 'Susilo', 'Sari', 'Jiang', 'Yin', 'Zhou', 'Shaaban',
    'Abdelgawad', 'Basnet', 'Dong', 'Abbott', 'Liu', 'Gad', 'Bhamare',
    'Churcher', 'Soe', 'Abbasi', 'Gu', 'Wu', 'Al-Omari', 'Omari', 'Habib'
]

def search_file(filepath):
    print(f"\n===== Searching in {filepath} =====")
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for idx, line in enumerate(lines):
        line_num = idx + 1
        found_keys = [k for k in keys if k in line]
        found_names = [n for n in names if re.search(r'\b' + re.escape(n) + r'\b', line, re.IGNORECASE)]
        
        if found_keys or found_names:
            # We want to print lines where a key is cited, or where one of the names appears
            # especially near a citation
            print(f"Line {line_num}: {line.strip()}")
            if found_keys:
                print(f"  -> Keys: {found_keys}")
            if found_names:
                print(f"  -> Names: {found_names}")

if __name__ == "__main__":
    search_file(r"g:\PycharmProjects\PythonProject\HLC\manuscript\manuscript.tex")
    search_file(r"g:\PycharmProjects\PythonProject\HLC\manuscript\supplementary.tex")
