# Load the package
library(RODBC) 

# Connect to MySQL (my credentials are mysql_server_64/root/.)
db = odbcConnect("mysql_server", uid="root", pwd="qpqp1234")
sqlQuery(db, "USE ma_charity_full")


# Extract calibration data from database
query = "SELECT a.contact_id,
	          c.recency AS 'recency',
	          c.seniority AS 'seniority',
	          c.active_duration AS 'active_duration',
	          c.counter AS 'frequency', 
	          c.avgamount AS 'avgamount',
	          c.maxamount as 'maxamount',
            c.nb_of_solitations AS 'nb_of_solitations',
            c.responsiveness AS 'responsiveness',
            e.summerdonation as 'summerdonation',
			      a.donation AS 'loyal',
			      AVG(a.amount) AS 'targetamount'
	          FROM ma_charity_full.assignment2 a
	              LEFT JOIN (SELECT contact_id, DATEDIFF(20180626, MAX(act_date))/365 AS recency, 
	                      DATEDIFF(20180626, MIN(act_date))/365 AS seniority,
	                      COUNT(DISTINCT LEFT(act_date,4)) AS 'active_duration', COUNT(amount) AS counter, 
	                      AVG(amount) AS avgamount, MAX(amount) AS maxamount, COUNT(campaign_id) AS nb_of_solitations,
	                      (COUNT(campaign_id)/COUNT(amount)) AS 'responsiveness'
                        FROM ma_charity_full.acts
                        WHERE (act_type_id = 'DO')
                        GROUP BY contact_id) AS c
				        ON c.contact_id = a.contact_id
				        LEFT JOIN (select contact_id, AVG(amount) AS 'summerdonation'
						    FROM ma_charity_full.acts
                        WHERE (MONTH(act_date) = 6 OR MONTH(act_date) = 7) AND 
                        ((YEAR(act_date) = 2017) OR (YEAR(act_date) = 2016) OR 
                        (YEAR(act_date) = 2015)) AND (act_type_id = 'DO')
                        GROUP BY contact_id) AS e
				        ON e.contact_id = a.contact_id
			      WHERE (a.calibration = 1)                  
            GROUP BY 1"

fulldata = sqlQuery(db, query)

# Show data
print(head(fulldata))

# Data Adjustments
fulldata$recency[fulldata$recency < 0.0027] <- max(fulldata$recency)
fulldata$seniority[fulldata$seniority <= 0.0027] <- 0.0027
fulldata$nb_of_solitations[fulldata$nb_of_solitations == 0] <- NA
fulldata$summerdonation[is.na(fulldata$summerdonation)] <- 0.0001
fulldata$responsiveness[fulldata$responsiveness <= 0.2500] <- 0.2500
fulldata$frequency[is.na(fulldata$frequency)] <- min(fulldata$frequency)

# Keeping no NA values
z2 = which(!is.na(fulldata$recency) & !is.na(fulldata$nb_of_solitations))
data_CV2 = fulldata[z2,1:12]
data_CV2$recency[data_CV2$recency > quantile(data_CV2$recency,0.9)] <- quantile(data_CV2$recency,0.9)
length(data_CV2$seniority[is.na(data_CV2$seniority)])

#spl = sample.split(data_CV2$loyal, 0.9)
#training = data_CV2[spl==TRUE, ]
#testing = data_CV2[spl==FALSE, ]

##################### LOGIT LASSO
library(glmnet)
full_x = cbind(data_CV2$recency, data_CV2$frequency, (data_CV2$recency * data_CV2$frequency), log(data_CV2$recency), log(data_CV2$frequency), log(data_CV2$avgamount), data_CV2$active_duration, log(data_CV2$summerdonation), (data_CV2$responsiveness))
#, (data_CV2$responsiveness*data_CV2$nb_of_solitations)
full_y = data_CV2$loyal

# Step 2: find the optimal value of lambda using cross-validation
cv.lasso = cv.glmnet(full_x, full_y, family = "binomial", alpha=1, type.measure = 'auc')
best.lambda = cv.lasso$lambda.min
fit = glmnet(full_x, full_y, family = "binomial", alpha=1, lambda = best.lambda, type.measure = 'auc')
fit$beta[,1]

# Lift chart
out = data.frame(contact_id = data_CV2$contact_id )
out$probs = predict(cv.lasso, full_x, s = best.lambda , type="response")

target = data_CV2$loyal[order(out$probs, decreasing=TRUE)] / sum(data_CV2$loyal)
gainchart = c(0, cumsum(target))

# Create a random selection sequence
random = seq(0, to = 1, length.out = length(data_CV2$loyal))

# Create the "perfect" selection sequence
perfect = data_CV2$loyal[order(data_CV2$loyal, decreasing=TRUE)] / sum(data_CV2$loyal)
perfect = c(0, cumsum(perfect))

# Plot gain chart, add random line
plot(gainchart)
lines(random)
lines(perfect)

# # Compute 1%, 5%, 10%, and 25% lift and improvement
q = c(0.01, 0.05, 0.10, 0.25)
x = quantile(gainchart, probs = q)

print("Lift:")
print(x/q)



#### Amount Model
z3 = which(!is.na(data_CV2$targetamount))

# Model 1
#amount.model = lm(formula = log(targetamount) ~ log(maxamount) + log(avgamount) + (nb_of_solitations*summerdonation),
#                   data = data_CV2[z3, ])
#+ responsiveness + nb_of_solitations + active_duration + sqrt(summerdonation)

# Model 2
amount_x = cbind(log(data_CV2[z3, ]$maxamount), log(data_CV2[z3, ]$avgamount), (data_CV2[z3, ]$nb_of_solitations * data_CV2[z3, ]$summerdonation))
amount_y = log(data_CV2[z3, ]$targetamount)

# Step 2: find the optimal value of lambda using cross-validation
cv.lasso_am = cv.glmnet(amount_x, amount_y, family = "gaussian", alpha= 1, type.measure = 'auc')
best.lambda_am = cv.lasso$lambda.min

fit = glmnet(amount_x, amount_y, family = "gaussian", alpha=1, lambda = best.lambda_am, type.measure = 'auc')
fit$beta[,1]

# Lift chart
#predict_am = exp(predict(object = amount.model, newdata = data_CV2[z3, ]))
predict_am = exp(predict(object = cv.lasso_am, amount_x, newdata = data_CV2[z3, ], s = best.lambda , type="response"))

target_am = data_CV2[z3, ]$targetamount[order(predict_am, decreasing=TRUE)] / sum(data_CV2[z3, ]$targetamount)
gainchart_am = c(0, cumsum(target_am))

# Create a random selection sequence
random_am = seq(0, to = 1, length.out = length(data_CV2[z3, ]$targetamount))

# Create the "perfect" selection sequence
perfect_am = data_CV2[z3, ]$targetamount[order(data_CV2[z3, ]$targetamount, decreasing=TRUE)] / sum(data_CV2[z3, ]$targetamount)
perfect_am = c(0, cumsum(perfect_am))

# Plot gain chart, add random line
plot(gainchart_am)
lines(random_am)
lines(perfect_am)

# # Compute 1%, 5%, 10%, and 25% lift and improvement
q_am = c(0.01, 0.05, 0.10, 0.25)
x_am = quantile(gainchart_am, probs = q_am)

print("Lift:")
print(x_am/q_am)


########################################
# MODEL
#### Data

sqlQuery(db, "USE ma_charity_full") 

# Extract calibration data from database
query2 = "SELECT a.contact_id,
	          c.recency AS 'recency',
	          c.seniority AS 'seniority',
	          c.active_duration AS 'active_duration',
	          c.counter AS 'frequency', 
	          c.avgamount AS 'avgamount',
	          c.maxamount as 'maxamount',
            c.nb_of_solitations AS 'nb_of_solitations',
            c.responsiveness AS 'responsiveness',
            e.summerdonation as 'summerdonation',
			      a.donation AS 'loyal',
			      AVG(a.amount) AS 'targetamount'
	          FROM ma_charity_full.assignment2 a
	              LEFT JOIN (SELECT contact_id, DATEDIFF(20180626, MAX(act_date))/365 AS recency, 
	                      ABS(DATEDIFF(20180626, MIN(act_date)))/365 AS seniority,
	                      COUNT(DISTINCT LEFT(act_date,4)) AS 'active_duration', COUNT(amount) AS counter, 
	                      AVG(amount) AS avgamount, MAX(amount) AS maxamount, COUNT(campaign_id) AS nb_of_solitations,
	                      (COUNT(campaign_id)/COUNT(amount)) AS 'responsiveness'
                        FROM ma_charity_full.acts
                        WHERE (act_type_id = 'DO')
                        GROUP BY contact_id) AS c
				        ON c.contact_id = a.contact_id
				        LEFT JOIN (select contact_id, AVG(amount) AS 'summerdonation'
						    FROM ma_charity_full.acts
                        WHERE (MONTH(act_date) = 6 OR MONTH(act_date) = 7) AND 
                        ((YEAR(act_date) = 2017) OR (YEAR(act_date) = 2016) OR 
                        (YEAR(act_date) = 2015)) AND (act_type_id = 'DO')
                        GROUP BY contact_id) AS e
				        ON e.contact_id = a.contact_id
			      WHERE (a.calibration = 0)                  
            GROUP BY 1"

unknowndata = sqlQuery(db, query2)

# Show data
print(head(unknowndata))

# Data Adjustments
unknowndata$recency[unknowndata$recency <= 0.0027] <- max(unknowndata$recency)
unknowndata$seniority[unknowndata$seniority <= 0.0027] <- 0.0027
unknowndata$nb_of_solitations[unknowndata$nb_of_solitations == 0] <- NA
unknowndata$summerdonation[is.na(unknowndata$summerdonation)] <- 0.0001
unknowndata$summerdonation[is.na(unknowndata$nb_of_solitations)] <- 0.0001
unknowndata$responsiveness[unknowndata$responsiveness <= 0.2500] <- 0.2500
unknowndata$frequency[is.na(unknowndata$frequency)] <- min(unknowndata$frequency)
unknowndata[c(3:9)][is.na(unknowndata[c(3:9)])] <- 0.001

# Keeping no NA values

#z8 = which(!is.na(unknowndata$nb_of_solitations))
data_CV3 = unknowndata[1:12]
data_CV3$recency[data_CV3$recency > quantile(data_CV3$recency[!is.na(data_CV3$recency)],0.9)] <- quantile(data_CV3$recency[!is.na(data_CV3$recency)],0.9)
data_CV3$recency[is.na(data_CV3$recency)] <- quantile(data_CV3$recency[!is.na(data_CV3$recency)],0.9)

unknown_test = cbind(data_CV3$recency, data_CV3$frequency, (data_CV3$recency * data_CV3$frequency), log(data_CV3$recency), log(data_CV3$frequency), log(data_CV3$avgamount), data_CV3$active_duration, log(data_CV3$summerdonation), (data_CV3$responsiveness))
amountt_x = cbind(log(data_CV3$maxamount), log(data_CV3$avgamount), (data_CV3$nb_of_solitations * data_CV3$summerdonation))

outt = data.frame(contact_id = data_CV3$contact_id )
outt$probs = predict(cv.lasso, unknown_test, lambda = best.lambda , type="response")
outt$probs[outt$probs <= 0.05] <- 0.01
#outt$probs = predict(cv.lasso, unknown_test, family = "binomial", alpha=1, lambda = best.lambda, type.measure = 'auc')
outt$amount = exp(predict(object = cv.lasso_am, amountt_x, newdata = data_CV3, s = best.lambda_am , type="response"))
outt$score  = outt$probs * outt$amount

z = which(outt$score > 2.5)
print(length(z))
#write.table(outt, "out.csv", row.names=FALSE, sep=",")

# Calculation for full profit
outt$profits <- outt$score - 2
summe = outt$profits[which(outt$profits > 0)]
res <- sum(summe)
print(res)

# Including column with 0 and 1
outt$solicitation = ifelse(outt$score > 2.5, 1, 0)
z11 = which(outt$solicitation == 1)
print(sum(outt$profits[outt$solicitation == 1]))
print(length(z11))

# Drop unwanted columns
out2 = subset(outt, select = -c(probs, amount, profits,score))

# Sorting contact_id
#install.packages('dplyr')
library('dplyr')
out3 <- arrange(out2, contact_id)

# Writing text file
write.table(out3, file = "Leyendecker_Patrick_Assignment_2.txt", append = FALSE, sep = "\t", dec = ".",
            row.names = FALSE, col.names = FALSE)

print(head(out3))

