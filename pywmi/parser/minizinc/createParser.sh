java -Xmx500M -cp "/usr/local/lib/antlr-4.7.1-complete.jar:$CLASSPATH" org.antlr.v4.Tool -visitor -Dlanguage=Python3 -o antlr/ minizinc.g4