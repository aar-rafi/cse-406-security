def remove_comments(input_file, output_file=None):
    """
    Removes single-line and multi-line comments from a Python file.

    Args:
        input_file (str): Path to the input Python file.
        output_file (str, optional): Path to the output file. If None,
                                     the changes are printed to the console.
    """
    with open(input_file, 'r') as f_in:
        lines = f_in.readlines()

    cleaned_lines = []
    in_multiline_comment = False

    for line in lines:
        stripped_line = line.strip()

        # Check for start of multi-line comment (docstrings at module/class/function level)
        # and ignore if it's part of a string
        if '"""' in stripped_line or "'''" in stripped_line:
            # Simple check, assumes `"""` or `'''` are on their own line or start/end of comment
            if stripped_line.count('"""') % 2 != 0 or stripped_line.count("'''") % 2 != 0:
                in_multiline_comment = not in_multiline_comment
                if in_multiline_comment: # If entering a multi-line comment, skip this line
                    continue
                else: # If exiting a multi-line comment, skip this line
                    continue
            else: # Handle cases where `"""` or `'''` might be within a line of code (e.g., string literal)
                # This is a basic approach and might not catch all edge cases
                if not in_multiline_comment and not stripped_line.startswith('#'):
                    cleaned_lines.append(line)
                continue


        if in_multiline_comment:
            continue

        # Remove single-line comments
        if stripped_line.startswith('#'):
            continue

        # Handle inline comments (e.g., `code # comment`)
        if '#' in line:
            code_part = line.split('#')[0].rstrip() # Get part before # and remove trailing whitespace
            if code_part: # Only add if there's actual code
                cleaned_lines.append(code_part + '\n') # Add newline back
            continue

        # Preserve empty lines and lines with only whitespace
        if line.strip() == '' and not stripped_line.startswith('#') and not in_multiline_comment:
            cleaned_lines.append(line)
        elif line.strip() != '':
            cleaned_lines.append(line)


    if output_file:
        with open(output_file, 'w') as f_out:
            f_out.writelines(cleaned_lines)
    else:
        for l in cleaned_lines:
            print(l, end='')

# Example usage:
# Assuming your file is named 'my_python_file.py'
# To print to console:
# remove_comments('my_python_file.py')

# To save to a new file:
# remove_comments('my_python_file.py', 'my_python_file_no_comments.py')

# To overwrite the original file (use with caution!):
import shutil
shutil.copyfile('ecc.py', 'ecc_backup.py') # Always backup first
remove_comments('ecc.py', 'ecc.py')