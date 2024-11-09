def convert_to_arff(input_file, output_file, relation_name):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Escribir el encabezado ARFF en el archivo de salida
    with open(output_file, 'w') as f:
        f.write(f"@relation {relation_name}\n\n")

        # Extraer los nombres de los atributos de la primera línea
        num_attributes = len(lines[0].split(',')) - 1  # Ignorar el último elemento que es la clase
        for i in range(num_attributes):
            f.write(f"@attribute attribute{i+1} numeric\n")

        # Escribir el atributo de clase
        class_attribute = "class"
        f.write(f"@attribute {class_attribute} {{0, 1}}\n\n")

        # Escribir los datos
        f.write("@data\n")
        for line in lines:
            f.write(line)

# Ejemplo de uso
input_file = "cleanDt1.txt"
output_file = "weka/Dt1-clean-data-Para-Weka.arff"
relation_name = "Dt1-cleand-data-Para-Weka"

convert_to_arff(input_file, output_file, relation_name)