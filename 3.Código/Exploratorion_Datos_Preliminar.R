install.packages("dplyr")

library(readxl)
library(dplyr)

#Carcar BD:
database <- read_excel("C:/Users/Cristina Oueghlani/Desktop/TFG/ALL_neurocloud_estudio_ALDAPA.xlsx", 
                        sheet = "normalized",
                        col_names = TRUE)
#View(database)
#head(database)


#_________________PDF DE LAS VARIABLES NO CONTINUAS--> HISTOGRAMAS y BOXPLOT________________

pdf(file = "VariablesContinuasHistBoxPlot.pdf")

par(mfrow = c(1,2)) #specify to save plots in 2x2 grid

hist(x=database$Age,
     xlab = "Edad de los pacientes",
     ylab = "",
     main = "Rango de edades",
     col = "lightblue",
     freq=FALSE)

#dnorm para calcular la función de densidad normal:
curve(dnorm(x, 
            mean=mean(database$Age), # Número o vector representando la/s media/s
            sd=sd(database$Age)), # Número o vector representando la/s desviación/es típica/s
      add=TRUE, col="black") 

boxplot(x=database$Age,
        xlab = "",
        ylab = "Edad de los pacientes",
        main = "Rango de edades",
        col = "lightblue")


variablesContinuas <- c(((select(database, -(1:8))) ) ) #length = 205
nombreColumnas <- c(names(database[0,-(1:8)]))

j<-1

for (i in variablesContinuas) {
  hist(i,
       main= list(nombreColumnas[c(j)] ),
       xlab = "",
       col = "lightblue",
       freq=FALSE
       )
  
  boxplot(i,
          col = "lightblue")
  j<-j+1
}

dev.off() #turn off PDF plotting


#_________________PDF DE LAS VARIABLES NO CONTINUAS--> HISTOGRAMAS + BOXPLOT________________

pdf(file = "MezclaHistogramaBoxplot.pdf")

par(mfrow = c(1,1)) #specify to save plots in 1x1 grid


hist(x=database$Age, 
     main = "Rango de edades",
     probability = TRUE, 
     axes = FALSE,#para encuadrar
     col = rgb(1, 0, 0, alpha = 0.5)
     )

axis(1) # Añade el eje horizontal
par(new = TRUE)

boxplot(x=database$Age, 
        horizontal = TRUE, #en posición horizontal 
        axes = FALSE, #para encuadrar
        lwd = 2, #grosor del margen
        col = rgb(0, 0, 0, alpha = 0.2))

variablesContinuas <- c(((select(database, -(1:8))) ) ) #length = 205
nombreColumnas <- c(names(database[0,-(1:8)]))

j<-1

for (i in variablesContinuas) {
  hist(i, 
       main= list(nombreColumnas[c(j)] ),
       probability = TRUE, 
       axes = FALSE,#para encuadrar
       col = rgb(1, 0, 0, alpha = 0.5))
  axis(1) # Añade el eje horizontal
  par(new = TRUE)
  boxplot(i, 
          horizontal = TRUE, #en posición horizontal 
          axes = FALSE, #para encuadrar
          lwd = 2, #grosor del margen
          col = rgb(0, 0, 0, alpha = 0.2))
  j<-j+1
}

dev.off() #turn off PDF plotting


#_____________________FRECUENCIAS DE LA COLUMNA C, E, F, G  y H_____________________________

#COLUMNA C: GÉNERO
#as.data.frame(table(database$Gender=='0'))
#Cantidad de 0's 
genderCero <- sum(database$Gender=='0')
genderCeroPorcentaje <- ((genderCero*100)/537)
print(paste("Cantidad de MUJERES:", genderCero, "-->", genderCeroPorcentaje, "%"))
#Cantidad de 1's 
genderUno <- sum(database$Gender=='1')
genderUnoPorcentaje <- ((genderUno*100)/537)
print(paste("Cantidad de HOMBRES:", genderUno, "-->", genderUnoPorcentaje, "%"))



#COLUMNA E: CSF_Result --> volumen cerebral dividido en muchas áreas. En caso de que
#                          algunos de los marcadores F, G o H hayan dado positivo, 
#                          la columna E marcará positivo
#Cantidad de 0's 
CSF_ResultCero <- sum(database$CSF_Result=='0')
genderCSF_ResultPorcentaje <- ((CSF_ResultCero*100)/537)
print(paste("Posibilidad de que NO se esté desarrollando alzheimer:", 
            CSF_ResultCero, "-->", genderCSF_ResultPorcentaje, "%"))
#Cantidad de 1's 
CSF_ResultUno <- sum(database$CSF_Result=='1')
genderCSF_ResultPorcentaje <- ((CSF_ResultUno*100)/537)
print(paste("Posibilidad de que SI se esté desarrollando alzheimer:", 
            CSF_ResultUno, "-->", genderCSF_ResultPorcentaje, "%"))
#Cantidad de NA's
CSF_ResultNA <-  sum(database$CSF_Result=='NA')
genderCSF_ResultPorcentaje <- ((CSF_ResultNA*100)/537)
print(paste("Gente que no se sabe si está desarrollando alzheimer:", 
            CSF_ResultNA, "-->", genderCSF_ResultPorcentaje, "%"))



#COLUMNA F: BETA AMILOIDES
#Cantidad de 0's 
CSF_ABCero <-sum(database$`CSF_AB42_Cutoff_LUD/Sant_pau`=='0')
CSF_ABCeroPorcentaje <- ((CSF_ABCero*100)/537)
print(paste("Cantidad de 0's:", CSF_ABCero, "-->", CSF_ABCeroPorcentaje, "%"))
#Cantidad de 1's 
CSF_ABUno <-sum(database$`CSF_AB42_Cutoff_LUD/Sant_pau`=='1')
CSF_ABCeroPorcentaje <- ((CSF_ABUno*100)/537)
print(paste("Cantidad de 1's:", CSF_ABUno, "-->", CSF_ABCeroPorcentaje, "%"))
#Cantidad de 9999's 
CSF_ABNueves <-sum(database$`CSF_AB42_Cutoff_LUD/Sant_pau`=='9999')
CSF_ABCeroPorcentaje <- ((CSF_ABNueves*100)/537)
print(paste("Cantidad de 9999's:", CSF_ABNueves, "-->",
            CSF_ABCeroPorcentaje, "%"))


#COLUMNA G: Proteina TAU 
#Cantidad de 0's 
CSF_TTAUCero <-sum(database$`CSF_TTAU_Cutoff_LUD/Sant_pau`=='0')
CSF_TTAUPorcentaje <- ((CSF_TTAUCero*100)/537)
print(paste("Cantidad de 0's:", CSF_TTAUCero, "-->", CSF_TTAUPorcentaje, "%"))
#Cantidad de 1's 
CSF_TTAUUno <-sum(database$`CSF_TTAU_Cutoff_LUD/Sant_pau`=='1')
CSF_TTAUPorcentaje <- ((CSF_TTAUUno*100)/537)
print(paste("Cantidad de 1's:", CSF_TTAUUno, "-->", CSF_TTAUPorcentaje, "%"))
#Cantidad de 9999's 
CSF_TTAUNueves <-sum(database$`CSF_TTAU_Cutoff_LUD/Sant_pau`=='9999')
CSF_TTAUPorcentaje <- ((CSF_TTAUNueves*100)/537)
print(paste("Cantidad de 9999's:", CSF_TTAUNueves, "-->", CSF_TTAUPorcentaje, "%"))


#COLUMNA H: Proteina TAU Fosforizada
#Cantidad de 0's 
CSF_PTAUCero <-sum(database$`CSF_PTAU_Cutoff_LUD/Sant_pau`=='0')
CSF_TTAUPorcentaje <- ((CSF_PTAUCero*100)/537)
print(paste("Cantidad de 0's:", CSF_PTAUCero, "-->", CSF_TTAUPorcentaje, "%"))
#Cantidad de 1's 
CSF_PTAUUno <-sum(database$`CSF_PTAU_Cutoff_LUD/Sant_pau`=='1')
CSF_TTAUPorcentaje <- ((CSF_PTAUUno*100)/537)
print(paste("Cantidad de 1's:", CSF_PTAUUno, "-->", CSF_TTAUPorcentaje, "%"))
#Cantidad de 9999's 
CSF_PTAUNueves <-sum(database$`CSF_PTAU_Cutoff_LUD/Sant_pau`=='9999')
CSF_TTAUPorcentaje <- ((CSF_TTAUNueves*100)/537)
print(paste("Cantidad de 9999's:", CSF_TTAUNueves, "-->", CSF_TTAUPorcentaje, "%"))


#______________________MATRIZ DE CORRELACIONES PEARSON: [1:6] con [1:6]___________________

install.packages("ggplot2")
install.packages("corrplot")

library(ggplot2)
library(corrplot)
library(readxl)

#Impotación de la BD pero sin la columna Project (función "cor" solo admite valores numéricos)
dbCor6 <- read_excel("C:/Users/Cristina Oueghlani/Desktop/TFG/ALL_neurocloud_estudio_ALDAPA.xlsx", 
                    sheet = "normalized", range = "C1:H537")
View(dbCor6)

sapply(dbCor6, class) #muestra el tipo de variables. Cor() --> 'x' must be numeric
dbCor6$CSF_Result <- as.numeric(dbCor6$CSF_Result) #combersión CSF_Result de "char" a "numeric"
sapply(dbCor6, class) #ahora CSF_Result es "numeric"

datosCorrelados6 <- cor(dbCor6, use="complete.obs") #COR. complete.obs: descarta toda la fila si NA está presente

#Exportación de los datos correlados a excel
#datosCorreladoss <- cor(databasecorrelation[,1:6], databasecorrelation[,7:208] , use="complete.obs") #208
write.csv2(datosCorrelados6,file="datosCorrelados_C-H.csv")

#round(datosCorrelados6, digits = 5)

col <- colorRampPalette(c("#BB4444", "#EE9988","#FFFFFF","#77AADD","#4477AA"))(10)

corrplot(datosCorrelados6, 
         #main = "Matriz de correlación C-H",
         method = "color", #shade(cuadrado), circle, color, number, pie, 
         col = col,
         addCoef.col = "black", #los números dentro del cuadrado en negro,
         number.cex = 0.75, #tamaño de los nº 
         order = "AOE", #tipo de ordenación AEO-> las + correlacionadas entre las -
         tl.cex = 0.5, #etiqueta tamaño
         tl.col = 'black', # etiquetas en negro
         tl.srt = 45, # etiquetas inclinadas 45º,
         cl.align.text = 'l', #ancho de los nº 1-(-1)
         )


#_____________________________MATRIZ DE CORRELACIONES PEARSON: [1:6] con [7:208] _______________________________

install.packages("corrplot")

library(corrplot)
library(readxl)


#Cargar la BD (modificada)
databasecorrelation <- read_excel("C:/Users/Cristina Oueghlani/Desktop/TFG/ALL_neurocloud_estudio_ALDAPA.xlsx", 
                        sheet = "normalizedCorrelation")
View(databasecorrelation)

databasecorrelation[databasecorrelation==9999] <-NA #Repleace "9999" to "NA"

sapply(databasecorrelation, class) #muestra el tipo de variables. Cor() --> 'x' must be numeric
databasecorrelation$CSF_Result <- as.numeric(databasecorrelation$CSF_Result) #combersión CSF_Result de "char" a "numeric"
sapply(databasecorrelation, class) #ahora CSF_Result es "numeric"
str(databasecorrelation) #otra forma de mostrar el tipo de variables y contenido de estas (comprobación CSF_Result="numeric")

pdf(file = "TablaDeCorrelaciones.pdf")

#Creación de la matriz de correlaciones entre [1:6] con [7:208], por intervalos de 10:
i=7
col <- colorRampPalette(c("#BB4444", "#EE9988","#FFFFFF","#77AADD","#4477AA"))(10)

while (i<198) {
  
  if(i<188){
    datosCorrelados <- cor(databasecorrelation[,1:6], databasecorrelation[,i:(i+9)] , use="complete.obs") #208
    round(datosCorrelados,2)
    
    corrplot(datosCorrelados,
             method = "color", #metodo de visualización 
             col = col,
             addCoef.col = "black", #los números dentro del cuadrado en negro
             number.cex = 0.75, #tamaño de los nº 
             #order = "AOE", #tipo de ordenación AEO-> las + correlacionadas entre las -
                             #Solo se puede usar en matrices cuadradas
             tl.cex = 0.5, #etiqueta tamaño
             tl.col = 'black', # etiquetas en negro
             tl.srt = 45, # etiquetas inclinadas 45º,
            )
    i=i+10

  }else{
    datosCorrelados <- cor(databasecorrelation[,1:6], databasecorrelation[,197:208] , use="complete.obs") #208
    round(datosCorrelados,2)
    
    corrplot(datosCorrelados,
             method = "color", #metodo de visualización 
             col = col,
             addCoef.col = "black", #los números dentro del cuadrado en negro
             number.cex = 0.75, #tamaño de los nº 
             tl.cex = 0.5, #etiqueta tamaño
             tl.col = 'black', # etiquetas en negro
             tl.srt = 45, # etiquetas inclinadas 45º,
            )
    break
  }
}
dev.off()

#Exportación de los datos correlados a excel
datosCorreladoss <- cor(databasecorrelation[,1:6], databasecorrelation[,7:208] , use="complete.obs") #208
write.csv2(datosCorreladoss,file="datosCorrelados.csv")