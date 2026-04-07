library(data.table)

## Takes in a bids file where the middle of each region is mu, and fourth column is a name
## will then provide a split set of LIET annotation files to the output dir w/ prefix ann_prefix

args <- commandArgs(trailingOnly = TRUE)
bids_file = args[1]
output_dir = args[2]
ann_prefix = args[3]

cat("\nUsing bid file", bids_file, 
    "\nSaving in", output_dir,
   "\nUsing ann_prefix", ann_prefix)

# read in the bids file
bids <- fread(bids_file)

# get number of files to be made if each has 1,450 annotations (must have at least 3 to ensure regions are sufficiently split)
num_files <- max(3, ceiling(nrow(bids)/1450))
# 
cat("\nNumber of files splitting annotation into", num_files)
cat("\n\tMinimum of 3 files to ensure regions are not overlapping")
# then have a repeating sequence of 1 to num_files to assign the sequences (with remainder accounted for)
remaining = nrow(bids)%%num_files
if (remaining > 0) {
    File <- c(rep(seq(1,num_files), nrow(bids)/num_files), seq(1,remaining))
} else {File <- rep(seq(1,num_files), nrow(bids)/num_files)}

bids$File <- File

# get mu and the "LIET end"
# bids$mu = as.integer((bids$V2+bids$V3)/2)
# bids$LIET_end = bids$mu + 100
# bids$mu = bids$V2
# bids$LIET_end = bids$V3
# length = bids$V5[1]

# now write each of the elements to a different file (should be sorted already)
cat("\nUsing fixed version")
for (num_file in seq(1,num_files)) {
    filt <- bids[bids$File == num_file,]
    cat("\nSaving file", num_file, nrow(filt))
    LIET_mu <- data.frame(data.table("chr"= filt$V1, 
                              "start"=filt$V2, 
                             "end"=filt$V3, 
                              "name"=filt$V4,
                              "length"=filt$V5,
                             "strand"=rep("+", nrow(filt))))
    write.table(LIET_mu, paste0(output_dir,"/", as.character(num_file),"_", ann_prefix, ".txt"), 
           row.names=FALSE, col.names=FALSE, sep="\t", quote=FALSE)
    
}
