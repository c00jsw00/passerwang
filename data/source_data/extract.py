# Read input file
with open("ASD_Release_202309_AS.txt", "r") as f:
    # Skip header
    next(f)
    
    # Open output file
    with open("okASD_Release_202309_AS.txt", "w") as output:
        # Write header
        output.write("target_id\ttarget_gene\torganism\tpdb_uniprot\tallosteric_pdb\tmodulator_serial\tmodulator_alias\tmodulator_chain\tmodulator_class\tmodulator_feature\tmodulator_name\tmodulator_resi\tfunction\tposition\tpubmed_id\tref_title\tsite_overlap\tallosteric_site_residue\n")
        
        # Process each line
        for line in f:
            fields = line.strip().split("\t")
            
            # Check if fields 4 (pdb), 6 (modulator), 7 (chain), 11 (mod_id) exist and are non-empty
            if len(fields) > 11 and all(fields[i] for i in [4, 6, 7, 11]):
                output.write(line)

print("Filtering complete. Results saved to new.txt")
