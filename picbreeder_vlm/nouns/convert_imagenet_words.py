import os

def main():
    output_dir = 'noun_lists'
    input_file = os.path.join(output_dir, 'imagenet21k_all_words.txt')
    output_file = os.path.join(output_dir, 'imagenet21k_all_first_words.txt')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            # Split by comma and take the first part
            first_word = line.split(',')[0].strip()
            if first_word:
                f_out.write(first_word + '\n')
    
    print(f"Processed {input_file} and created {output_file}")

if __name__ == "__main__":
    main()
