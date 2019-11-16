#sh pipeline.sh -D gaussian -C 0.8 -S asia -M cpc --compute scores --replot --score skeleton;

sh pipeline.sh -D gaussian -C 0.8 -S alarm -M cpc --compute scores --replot --score all;
sh pipeline.sh -D student -C 0.8 -S alarm -M cpc --compute scores --replot --score all;
sh pipeline.sh -D dirichlet -C 0.8 -S alarm -M cpc --compute scores --replot --score all;

#sh pipeline.sh -D gaussian -C 0.8 -S asia -M elidan --compute scores --replot --score all;
#sh pipeline.sh -D student -C 0.8 -S asia -M elidan --compute scores --replot --score all;
#sh pipeline.sh -D dirichlet -C 0.8 -S asia -M elidan --compute scores --replot --score all;
#sh pipeline.sh -D gaussian -C 0.8 -S alarm -M elidan --compute scores --recompute --replot --score skeleton;
#sh pipeline.sh -D gaussian -C 0.8 -S asia -M elidan --compute scores --replot --score skeleton;
#sh pipeline.sh -D gaussian -C 0.8 -S asia -M elidan --compute scores --replot;

#sh pipeline.sh -D dirichlet -C 0.8 -S asia -M cpc --compute scores --replot;
#sh pipeline.sh -D dirichlet -C 0.8 -S asia -M elidan --compute scores --replot;

#sh pipeline.sh -D student -C 0.8 -S asia -M cpc --compute scores --replot;
#sh pipeline.sh -D student -C 0.8 -S asia -M elidan --compute scores --replot;

#sh pipeline.sh -D gaussian -C 0.8 -S asia -M elidan --compute time --replot;
