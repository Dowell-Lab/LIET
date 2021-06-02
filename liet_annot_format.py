import argparse
import sys

description_text = ("Extracts extreme coordinates of annotation for input "
    "list of gene IDs and .gtf annotation file.")

gene_id_text = ("Path to text file containing list of gene IDs")

annot_text = ("Path to .gtf file containing annotations. Must contain exon "
    "features as well as 'gene_id' keyword in last column")

parser = argparse.ArgumentParser(description = description_text)

# INPUT GENE LIST
parser.add_argument(
    '-i', '--ids',
    type = str,
    help = gene_id_text
)

parser.add_argument(
    '-a', '--annot',
    type = str,
    help = annot_text
)

args = parser.parse_args()

# Read in all the gene IDs
#gene_ids = []
#with open(args.ids, 'r') as id_file:
#    gene_ids = [id.strip() for id in id_file]
#gene_ids = set(gene_ids)

# Read in all the genes and the padded regions

gene_pad = {}

with open(args.ids, 'r') as reg_file:
    for reg in reg_file:

        chrom, pstart, pstop, gid = reg.strip().split('\t')
        pstart = int(pstart)
        pstop = int(pstop)

        gene_pad[gid] = (chrom, pstart, pstop)


gene_ids = set(gene_pad.keys())

# Initialize annotation dictionary
annots = {id:{'chrom': set(), 'strand': set(), 'exons':[]} for id in gene_ids}
#print(annots, file=sys.stdout)

# Chromosomes present in annotations
chrom_set = set()

# Read full gtf and populate annots dictionary with chrom/strand/exon coords
with open(args.annot, 'r') as annot_file:
    for line in annot_file:

        line = line.strip().split('\t')

        # Parse info column (currently only using 'gene_id' field)
        info = line[8].strip().strip(';').split(';')
        info_list = [i.strip().split(' ') for i in info]
        info_dict = {i[0]:i[1].strip('"') for i in info_list}

        # Check if line's gene ID is in query list
        if info_dict['gene_id'] in gene_ids:
            
            # Check if feature type is "exon"
            feat = line[2].strip()
            if feat == 'exon':
                
                # Append chrom and strand strings
                id = info_dict['gene_id']
                chrom = line[0].strip()
                strand = line[6].strip()
                annots[id]['chrom'].add(chrom)
                annots[id]['strand'].add(strand)

                # Tracking all unique chromosomes
                chrom_set.add(chrom)

                # Append exon coordinates
                start = int(line[3].strip())
                stop = int(line[4].strip())
                annots[id]['exons'].append((start, stop))

            else:
                continue
        else:
            continue


# Inverted dictionary used to aid in sorting prior to final print
# format: {'chr#': {(start, stop, '+/-'): 'gene_id', ...}, ...}
bed_form = {chrom: {} for chrom in chrom_set}

# With <annots> dict complete, consolide exons and write it to <bed_form>
for id, annot in annots.items():

    chrom = annot['chrom']

    # Write annotation to <bed_form> if exons came from exactly one chromosome
    if len(chrom) == 1:

        chrom = chrom.pop()
        exon_bounds = list(zip(*annot['exons']))
        strand_str = ','.join(sorted(annot['strand']))
        coord_tup = (min(exon_bounds[0]), max(exon_bounds[1]), strand_str)
        
        bed_form[chrom][coord_tup] = id

    # If # of chromosomes is zero or greater than 1 write error line
    else:

        err_string = '\t'.join(['chr0', '0', '0', id])
        print(err_string, file=sys.stderr)

# Loop over <bed_form> sorted by chromosome order
for chrom in sorted(bed_form.keys(), key = lambda x: int(x.strip('chr'))):

    # Loop over regions, sorted by their coordinates)
    for region in sorted(bed_form[chrom].keys()):
        
        # Format entry into a bed file format (chr \t start \t stop \t id)
        gid = bed_form[str(chrom)][region]
        
        # Calculate padding, if provided in input
        pstart = gene_pad[gid][1]
        pstop = gene_pad[gid][2]

        gstart = region[0]
        gstop = region[1]
        gstrand = region[2]

        if gstrand == '+':
            pad5 = str(int(abs(pstart - gstart)))
            pad3 = str(int(abs(pstop - gstop)))
        elif gstrand == '-':
            pad5 = str(int(abs(pstop - gstop)))
            pad3 = str(int(abs(pstart - gstart)))
        else:
            continue

        # Format new bed line containing gene coordinates and padding
        gstart = str(gstart)
        gstop = str(gstop)
        pad = ','.join([pad5, pad3])
        bed_line = [chrom, gstart, gstop, gstrand, gid, pad]
        bed_line = '\t'.join(bed_line)

        # Print annotation line to standard out
        print(bed_line, file=sys.stdout)