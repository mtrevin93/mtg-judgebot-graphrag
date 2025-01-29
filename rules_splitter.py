def split_rules_into_files(rules_text, output_dir="input"):
    """
    Split rules text into separate files based on rule numbers, including section headers
    and grouping lettered subrules with their parent.
    """
    import os
    import re
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store section headers
    section_headers = {}
    # Dictionary to store rule groups
    rule_groups = {}
    
    current_section = None
    current_rule = None
    current_parent_rule = None
    current_text = []
    
    # Split the text into lines and process
    for line in rules_text.split('\n'):
        # Check for main section header (e.g., "904. Archenemy")
        section_match = re.match(r'^(\d+)\.\s+(.+)$', line)
        if section_match:
            current_section = section_match.group(1)
            section_headers[current_section] = line
            continue
            
        # Check for parent rule (e.g., "903.13. Commander Draft")
        parent_match = re.match(r'^(\d+\.\d+)\.\s+(.+)$', line)
        if parent_match:
            # Save previous rule group if it exists
            if current_rule is not None:
                rule_groups[current_rule] = '\n'.join(filter(None, current_text))  # Remove empty strings
            
            current_parent_rule = parent_match.group(1)
            current_rule = current_parent_rule
            section = current_parent_rule.split('.')[0]
            current_text = [section_headers.get(section, ''), '', line]
            continue
            
        # Check for lettered rule (e.g., "903.13a")
        rule_match = re.match(r'^(\d+\.\d+)[a-z]\s+(.+)$', line)
        if rule_match:
            base_rule = rule_match.group(1)
            if base_rule != current_parent_rule:
                # Save previous rule group if it exists
                if current_rule is not None:
                    rule_groups[current_rule] = '\n'.join(filter(None, current_text))  # Remove empty strings
                current_parent_rule = base_rule
                current_rule = base_rule
                section = base_rule.split('.')[0]
                current_text = [section_headers.get(section, ''), '', line]
            else:
                current_text.append(line)
        elif line.strip() and current_rule is not None:
            current_text.append(line)
    
    # Save last rule group
    if current_rule is not None:
        rule_groups[current_rule] = '\n'.join(filter(None, current_text))  # Remove empty strings
    
    # Write files
    for rule_num, rule_text in rule_groups.items():
        # Clean up multiple newlines and remove leading/trailing whitespace
        cleaned_text = re.sub(r'\n{3,}', '\n\n', rule_text.strip())
        filename = f"{output_dir}/rule_{rule_num.replace('.', '_')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(cleaned_text + '\n')  # Add single newline at end of file

# Read and process the rules
with open('raw_input/rules.txt', 'r', encoding='utf-8') as f:
    rules_text = f.read()

split_rules_into_files(rules_text)