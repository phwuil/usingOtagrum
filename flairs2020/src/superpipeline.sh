sh pipeline.sh -D gaussian -C 0.8 -S asia -M cpc --compute loglikelihood;
sh pipeline.sh -D gaussian -C 0.8 -S asia -M elidan --compute loglikelihood;

sh pipeline.sh -D dirichlet -C 0.8 -S asia -M cpc --compute loglikelihood;
sh pipeline.sh -D dirichlet -C 0.8 -S asia -M elidan --compute loglikelihood;

sh pipeline.sh -D student -C 0.8 -S asia -M cpc --compute loglikelihood;
sh pipeline.sh -D student -C 0.8 -S asia -M elidan --compute loglikelihood;
