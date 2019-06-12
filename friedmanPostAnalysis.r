if(!require(foreach))
{
	print("You are missing the package 'foreach', we will now try to install it...")
	install.packages("foreach")
	#library(foreach)
}

if(!require(xtable))
{
	print("You are missing the package 'xtable', we will now try to install it...")
	install.packages("xtable")
	#library(xtable)
}

if(!require(igraph))
{
	print("You are missing the package 'igraph', we will now try to install it...")
	install.packages("igraph")
	#library(igraph)
}

library(coin)
library(multcomp)
library(colorspace)
library(foreach)
library(xtable)
library(igraph)

labels <- strsplit(row.names(output$PostHoc.Test), " - ")
matrix <- matrix(nrow = length(methods), ncol = length(methods), dimnames=list(methods, methods))
adjMatrix <- matrix #adjacency matrix

alpha <- 0.05

raw_ranks <- c()
foreach(p = 1:length(problems)) %do% {
    indexes <- seq((p-1)*(length(methods)-1) + p, p*(length(methods)-1) + p)
    p_ranks <- rank(Data$Table[indexes])
    raw_ranks <- rbind(raw_ranks, p_ranks)
}
# ranks consists of raw ranks now
#print(ranks)

# aggregate ranks
ranks = c()
foreach(m = 1:length(methods)) %do% {
    ranks[m] <- mean(raw_ranks[,m])
}
print(paste("Ranks: ", ranks))

foreach(i = 1:length(output$PostHoc.Test)) %do% {
	l1 <- unlist(labels[i])[1]
	l2 <- unlist(labels[i])[2]
	#print(paste("label1: ", l1, " label2: ", l2))
	
	l1Idx <- match(l1, methods)[1]
	l2Idx <- match(l2, methods)[1]
	#avg1 <- sum(Data$Table[seq(l1Idx, length(Data$Table), length(methods))]) / length(problems)
	#avg2 <- sum(Data$Table[seq(l2Idx, length(Data$Table), length(methods))]) / length(problems)
	
	#print(paste("avg1: ", avg1, " avg2: ", avg2))
	
	if (ranks[l1Idx] < ranks[l2Idx]) {
		if (output$PostHoc.Test[i] < alpha) {
			matrix[l1, l2] <- sprintf("\\textbf{%.3f}", output$PostHoc.Test[i])
			adjMatrix[l1, l2] <- 1
		}
		else
		{
			matrix[l1, l2] <- sprintf("%.3f", output$PostHoc.Test[i])
		}
	} else {
		if (output$PostHoc.Test[i] < alpha) {
			matrix[l2, l1] <- sprintf("\\textbf{%.3f}", output$PostHoc.Test[i])
			adjMatrix[l2, l1] <- 1
		}
		else
		{
			matrix[l2, l1] <- sprintf("%.3f", output$PostHoc.Test[i])
		}
	}
}

#print(matrix)
print(xtable(matrix, digits = 3), type="latex", sanitize.text.function = function(x){x})

graph <- graph.adjacency(adjMatrix)
V(graph)$label <- methods
V(graph)$label.cex <- 2 #font size
V(graph)$color <- "Lightgray"

#reset margin
par(mar=c(0,0,0,0))
plot(graph, layout=layout.circle, vertex.size=75, edge.color="Black") 

