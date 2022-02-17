library(rsconnect)
library(shiny)

rsconnect::setAccountInfo(
	name='alecrsf',
	token='6604B43F7F4511384762B5A4031832FD',
	secret='AysijiDEKxupO7FJQpUYP9JWS6OTt0eVDufu9f/s')

library(httr)    
set_config(use_proxy(url="10.3.100.207",port=8080))
rsconnect::deployApp('deploy/HR_Analytics.Rmd')
