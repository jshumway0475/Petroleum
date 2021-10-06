# Petroleum Discounted Cash Flow Analysis  
This repository contains a Jupyter Notebook called **Multi-Well DCF.ipynb** that takes as input oil and natural gas price forecasts (example file provided STR-071521.csv) and a list of properties that could be existing wells or future wells (example file provided property_table.csv) with several fields needed to create forecasts for oil and gas production, expenses, and capital. The Jupyter Notebook generates the forecasts for the time frame specified, discounts the projected cash flow stream, and provides output of key economic metrics such as net present value, rate of return, etc.

Also included is a Jupyter Notebook called **DCA Calcs.ipynb** that is a simple calculator to forecast oil and gas volumes from Arps equations. This is a primary component of the Muti-Well DCF.ipynb program and is provided for reference.  

It is important to note that all properties in the property_table.csv file must have forecast parameters anchored to a common start month ('Base Date' in cell 10 of the Multi-Well DCF.ipynb program). In the example file, the base date (start month) is set to July 2021.
