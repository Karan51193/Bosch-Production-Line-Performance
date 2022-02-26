from wsgiref import simple_server
from flask import Flask, request, render_template,make_response
from flask import Response,send_file
import os
from flask_cors import CORS, cross_origin
import flask_monitoringdashboard as dashboard
import json
import pickle
import pandas as pd
import joblib
import sklearn
import xgboost as xgb
import gc
import pre_process_num
import pre_process_date
import os
import csv


print(xgb.__version__)

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
dashboard.bind(app)
CORS(app)


@app.route("/", methods=['GET'])
# @app.route('/<path:path>')
@cross_origin()
def home():
    # file_path = request.form.get("file_path")
    file=request.files.get("file")
    # file=request.files['csvfile']
    # # print('Home path',file_path)
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():

    try:
        if request.json is not None:
            file=request.files.get("file")
            # print('path',file_path)

        elif request.form is not None:

            num_fea=['Id', 'L3_S38_F3960', 'L3_S38_F3956', 'L3_S38_F3952', 'L3_S32_F3850', 'L3_S30_F3554', 'L3_S29_F3442', 'L3_S30_F3754', 'L3_S30_F3774', 'L3_S35_F3896', 'L3_S30_F3749', 'L3_S30_F3574', 'L1_S24_F1723', 'L3_S29_F3330', 'L3_S30_F3819', 'L3_S30_F3519', 'L3_S33_F3865', 'L1_S24_F1846', 'L3_S30_F3794', 'L3_S29_F3479', 'L3_S30_F3809', 'L1_S24_F1695', 'L3_S29_F3333', 'L3_S30_F3804', 'L3_S29_F3449', 'L3_S33_F3859', 'L3_S29_F3336', 'L3_S29_F3430', 'L3_S29_F3424', 'L3_S30_F3769', 'L3_S30_F3759', 'L3_S29_F3382', 'L3_S29_F3436', 'L3_S29_F3370', 'L3_S29_F3373', 'L3_S29_F3354', 'L3_S29_F3427', 'L3_S30_F3829', 'L3_S29_F3315', 'L0_S1_F28', 'L3_S29_F3395', 'L3_S30_F3744', 'L3_S30_F3499', 'L3_S30_F3534', 'L3_S29_F3324', 'L3_S36_F3924', 'L3_S29_F3401', 'L3_S29_F3327', 'L3_S30_F3569', 'L3_S33_F3863', 'L3_S30_F3784', 'L3_S29_F3345', 'L3_S30_F3609', 'L3_S29_F3376', 'L3_S29_F3367', 'L3_S36_F3920', 'L3_S33_F3855', 'L3_S30_F3799', 'L3_S29_F3388', 'L0_S0_F20', 'L3_S30_F3539', 'L3_S29_F3318', 'L0_S0_F0', 'L3_S30_F3494', 'L3_S30_F3604', 'L3_S30_F3764', 'L3_S33_F3857', 'L3_S30_F3709', 'L3_S29_F3379', 'L3_S30_F3559', 'L3_S30_F3514', 'L3_S30_F3639', 'L3_S30_F3669', 'L3_S29_F3452', 'L3_S30_F3579', 'L0_S0_F2', 'L3_S30_F3629', 'L3_S30_F3649', 'L3_S29_F3342', 'L3_S29_F3455', 'L0_S9_F160', 'L3_S29_F3458', 'L3_S30_F3544', 'L0_S0_F18', 'L3_S30_F3589', 'L3_S29_F3439', 'L3_S29_F3351', 'L3_S29_F3348', 'L3_S30_F3689', 'L0_S11_F290', 'L3_S29_F3404']
            date_fea=['Id', 'L0_S0_D1', 'L0_S0_D3', 'L0_S0_D5', 'L0_S0_D7', 'L0_S0_D9', 'L0_S0_D11', 'L0_S0_D13', 'L0_S0_D15', 'L0_S0_D17', 'L0_S0_D19', 'L0_S0_D21', 'L0_S0_D23', 'L0_S1_D26', 'L0_S1_D30', 'L0_S2_D34', 'L0_S2_D38', 'L0_S2_D42', 'L0_S2_D46', 'L0_S2_D50', 'L0_S2_D54', 'L0_S2_D58', 'L0_S2_D62', 'L0_S2_D66', 'L0_S3_D70', 'L0_S3_D74', 'L0_S3_D78', 'L0_S3_D82', 'L0_S3_D86', 'L0_S3_D90', 'L0_S3_D94', 'L0_S3_D98', 'L0_S3_D102', 'L0_S4_D106', 'L0_S4_D111', 'L0_S5_D115', 'L0_S5_D117', 'L0_S6_D120', 'L0_S6_D124', 'L0_S6_D127', 'L0_S6_D130', 'L0_S6_D134', 'L0_S7_D137', 'L0_S7_D139', 'L0_S7_D140', 'L0_S7_D141', 'L0_S7_D143', 'L0_S8_D145', 'L0_S8_D147', 'L0_S8_D148', 'L0_S8_D150', 'L0_S9_D152', 'L0_S9_D157', 'L0_S9_D162', 'L0_S9_D167', 'L0_S9_D172', 'L0_S9_D177', 'L0_S9_D182', 'L0_S9_D187', 'L0_S9_D192', 'L0_S9_D197', 'L0_S9_D202', 'L0_S9_D207', 'L0_S9_D212', 'L0_S10_D216', 'L0_S10_D221', 'L0_S10_D226', 'L0_S10_D231', 'L0_S10_D236', 'L0_S10_D241', 'L0_S10_D246', 'L0_S10_D251', 'L0_S10_D256', 'L0_S10_D261', 'L0_S10_D266', 'L0_S10_D271', 'L0_S10_D276', 'L0_S11_D280', 'L0_S11_D284', 'L0_S11_D288', 'L0_S11_D292', 'L0_S11_D296', 'L0_S11_D300', 'L0_S11_D304', 'L0_S11_D308', 'L0_S11_D312', 'L0_S11_D316', 'L0_S11_D320', 'L0_S11_D324', 'L0_S11_D328', 'L0_S12_D331', 'L0_S12_D333', 'L0_S12_D335', 'L0_S12_D337', 'L0_S12_D339', 'L0_S12_D341', 'L0_S12_D343', 'L0_S12_D345', 'L0_S12_D347', 'L0_S12_D349', 'L0_S12_D351', 'L0_S12_D353', 'L0_S13_D355', 'L0_S13_D357', 'L0_S14_D360', 'L0_S14_D364', 'L0_S14_D368', 'L0_S14_D372', 'L0_S14_D376', 'L0_S14_D380', 'L0_S14_D384', 'L0_S14_D388', 'L0_S14_D392', 'L0_S15_D395', 'L0_S15_D398', 'L0_S15_D401', 'L0_S15_D404', 'L0_S15_D407', 'L0_S15_D410', 'L0_S15_D413', 'L0_S15_D416', 'L0_S15_D419', 'L0_S16_D423', 'L0_S16_D428', 'L0_S17_D432', 'L0_S17_D434', 'L0_S18_D437', 'L0_S18_D441', 'L0_S18_D444', 'L0_S18_D447', 'L0_S18_D451', 'L0_S19_D454', 'L0_S19_D456', 'L0_S19_D457', 'L0_S19_D458', 'L0_S19_D460', 'L0_S20_D462', 'L0_S20_D464', 'L0_S20_D465', 'L0_S20_D467', 'L0_S21_D469', 'L0_S21_D474', 'L0_S21_D479', 'L0_S21_D484', 'L0_S21_D489', 'L0_S21_D494', 'L0_S21_D499', 'L0_S21_D504', 'L0_S21_D509', 'L0_S21_D514', 'L0_S21_D519', 'L0_S21_D524', 'L0_S21_D529', 'L0_S21_D534', 'L0_S21_D539', 'L0_S22_D543', 'L0_S22_D548', 'L0_S22_D553', 'L0_S22_D558', 'L0_S22_D563', 'L0_S22_D568', 'L0_S22_D573', 'L0_S22_D578', 'L0_S22_D583', 'L0_S22_D588', 'L0_S22_D593', 'L0_S22_D598', 'L0_S22_D603', 'L0_S22_D608', 'L0_S22_D613', 'L0_S23_D617', 'L0_S23_D621', 'L0_S23_D625', 'L0_S23_D629', 'L0_S23_D633', 'L0_S23_D637', 'L0_S23_D641', 'L0_S23_D645', 'L0_S23_D649', 'L0_S23_D653', 'L0_S23_D657', 'L0_S23_D661', 'L0_S23_D665', 'L0_S23_D669', 'L0_S23_D673', 'L1_S24_D677', 'L1_S24_D681', 'L1_S24_D685', 'L1_S24_D689', 'L1_S24_D693', 'L1_S24_D697', 'L1_S24_D702', 'L1_S24_D707', 'L1_S24_D712', 'L1_S24_D716', 'L1_S24_D721', 'L1_S24_D725', 'L1_S24_D730', 'L1_S24_D735', 'L1_S24_D739', 'L1_S24_D743', 'L1_S24_D748', 'L1_S24_D753', 'L1_S24_D758', 'L1_S24_D763', 'L1_S24_D768', 'L1_S24_D772', 'L1_S24_D777', 'L1_S24_D782', 'L1_S24_D787', 'L1_S24_D792', 'L1_S24_D797', 'L1_S24_D801', 'L1_S24_D804', 'L1_S24_D807', 'L1_S24_D809', 'L1_S24_D811', 'L1_S24_D813', 'L1_S24_D815', 'L1_S24_D818', 'L1_S24_D822', 'L1_S24_D826', 'L1_S24_D831', 'L1_S24_D836', 'L1_S24_D841', 'L1_S24_D846', 'L1_S24_D850', 'L1_S24_D854', 'L1_S24_D859', 'L1_S24_D864', 'L1_S24_D869', 'L1_S24_D874', 'L1_S24_D879', 'L1_S24_D884', 'L1_S24_D889', 'L1_S24_D894', 'L1_S24_D899', 'L1_S24_D904', 'L1_S24_D909', 'L1_S24_D913', 'L1_S24_D917', 'L1_S24_D922', 'L1_S24_D927', 'L1_S24_D932', 'L1_S24_D937', 'L1_S24_D941', 'L1_S24_D945', 'L1_S24_D950', 'L1_S24_D955', 'L1_S24_D960', 'L1_S24_D965', 'L1_S24_D970', 'L1_S24_D975', 'L1_S24_D980', 'L1_S24_D985', 'L1_S24_D990', 'L1_S24_D995', 'L1_S24_D999', 'L1_S24_D1001', 'L1_S24_D1003', 'L1_S24_D1005', 'L1_S24_D1007', 'L1_S24_D1009', 'L1_S24_D1011', 'L1_S24_D1013', 'L1_S24_D1015', 'L1_S24_D1018', 'L1_S24_D1023', 'L1_S24_D1028', 'L1_S24_D1033', 'L1_S24_D1038', 'L1_S24_D1043', 'L1_S24_D1048', 'L1_S24_D1053', 'L1_S24_D1058', 'L1_S24_D1062', 'L1_S24_D1066', 'L1_S24_D1070', 'L1_S24_D1074', 'L1_S24_D1077', 'L1_S24_D1081', 'L1_S24_D1085', 'L1_S24_D1089', 'L1_S24_D1092', 'L1_S24_D1096', 'L1_S24_D1100', 'L1_S24_D1104', 'L1_S24_D1108', 'L1_S24_D1112', 'L1_S24_D1116', 'L1_S24_D1120', 'L1_S24_D1124', 'L1_S24_D1128', 'L1_S24_D1132', 'L1_S24_D1135', 'L1_S24_D1138', 'L1_S24_D1141', 'L1_S24_D1143', 'L1_S24_D1146', 'L1_S24_D1149', 'L1_S24_D1151', 'L1_S24_D1153', 'L1_S24_D1155', 'L1_S24_D1158', 'L1_S24_D1163', 'L1_S24_D1168', 'L1_S24_D1171', 'L1_S24_D1173', 'L1_S24_D1175', 'L1_S24_D1178', 'L1_S24_D1182', 'L1_S24_D1186', 'L1_S24_D1190', 'L1_S24_D1194', 'L1_S24_D1199', 'L1_S24_D1204', 'L1_S24_D1209', 'L1_S24_D1214', 'L1_S24_D1218', 'L1_S24_D1222', 'L1_S24_D1227', 'L1_S24_D1232', 'L1_S24_D1237', 'L1_S24_D1242', 'L1_S24_D1247', 'L1_S24_D1252', 'L1_S24_D1257', 'L1_S24_D1262', 'L1_S24_D1267', 'L1_S24_D1272', 'L1_S24_D1277', 'L1_S24_D1281', 'L1_S24_D1285', 'L1_S24_D1290', 'L1_S24_D1295', 'L1_S24_D1300', 'L1_S24_D1305', 'L1_S24_D1309', 'L1_S24_D1313', 'L1_S24_D1318', 'L1_S24_D1323', 'L1_S24_D1328', 'L1_S24_D1333', 'L1_S24_D1338', 'L1_S24_D1343', 'L1_S24_D1348', 'L1_S24_D1353', 'L1_S24_D1358', 'L1_S24_D1363', 'L1_S24_D1368', 'L1_S24_D1373', 'L1_S24_D1378', 'L1_S24_D1383', 'L1_S24_D1388', 'L1_S24_D1393', 'L1_S24_D1398', 'L1_S24_D1403', 'L1_S24_D1408', 'L1_S24_D1413', 'L1_S24_D1418', 'L1_S24_D1423', 'L1_S24_D1428', 'L1_S24_D1433', 'L1_S24_D1438', 'L1_S24_D1443', 'L1_S24_D1448', 'L1_S24_D1453', 'L1_S24_D1457', 'L1_S24_D1461', 'L1_S24_D1465', 'L1_S24_D1469', 'L1_S24_D1472', 'L1_S24_D1476', 'L1_S24_D1480', 'L1_S24_D1484', 'L1_S24_D1488', 'L1_S24_D1492', 'L1_S24_D1496', 'L1_S24_D1500', 'L1_S24_D1504', 'L1_S24_D1508', 'L1_S24_D1511', 'L1_S24_D1513', 'L1_S24_D1515', 'L1_S24_D1517', 'L1_S24_D1519', 'L1_S24_D1522', 'L1_S24_D1527', 'L1_S24_D1532', 'L1_S24_D1536', 'L1_S24_D1541', 'L1_S24_D1546', 'L1_S24_D1550', 'L1_S24_D1554', 'L1_S24_D1558', 'L1_S24_D1562', 'L1_S24_D1566', 'L1_S24_D1568', 'L1_S24_D1570', 'L1_S24_D1572', 'L1_S24_D1574', 'L1_S24_D1576', 'L1_S24_D1579', 'L1_S24_D1583', 'L1_S24_D1587', 'L1_S24_D1591', 'L1_S24_D1596', 'L1_S24_D1601', 'L1_S24_D1606', 'L1_S24_D1611', 'L1_S24_D1615', 'L1_S24_D1619', 'L1_S24_D1624', 'L1_S24_D1629', 'L1_S24_D1634', 'L1_S24_D1639', 'L1_S24_D1644', 'L1_S24_D1649', 'L1_S24_D1654', 'L1_S24_D1659', 'L1_S24_D1664', 'L1_S24_D1669', 'L1_S24_D1674', 'L1_S24_D1678', 'L1_S24_D1682', 'L1_S24_D1687', 'L1_S24_D1692', 'L1_S24_D1697', 'L1_S24_D1702', 'L1_S24_D1706', 'L1_S24_D1710', 'L1_S24_D1715', 'L1_S24_D1720', 'L1_S24_D1725', 'L1_S24_D1730', 'L1_S24_D1735', 'L1_S24_D1740', 'L1_S24_D1745', 'L1_S24_D1750', 'L1_S24_D1755', 'L1_S24_D1760', 'L1_S24_D1765', 'L1_S24_D1770', 'L1_S24_D1775', 'L1_S24_D1780', 'L1_S24_D1785', 'L1_S24_D1790', 'L1_S24_D1795', 'L1_S24_D1800', 'L1_S24_D1805', 'L1_S24_D1809', 'L1_S24_D1811', 'L1_S24_D1813', 'L1_S24_D1815', 'L1_S24_D1817', 'L1_S24_D1819', 'L1_S24_D1821', 'L1_S24_D1823', 'L1_S24_D1825', 'L1_S24_D1826', 'L1_S24_D1828', 'L1_S24_D1830', 'L1_S24_D1832', 'L1_S24_D1833', 'L1_S24_D1835', 'L1_S24_D1837', 'L1_S24_D1839', 'L1_S24_D1841', 'L1_S24_D1843', 'L1_S24_D1845', 'L1_S24_D1847', 'L1_S24_D1849', 'L1_S24_D1851', 'L1_S25_D1854', 'L1_S25_D1857', 'L1_S25_D1860', 'L1_S25_D1862', 'L1_S25_D1864', 'L1_S25_D1867', 'L1_S25_D1871', 'L1_S25_D1875', 'L1_S25_D1879', 'L1_S25_D1883', 'L1_S25_D1887', 'L1_S25_D1891', 'L1_S25_D1893', 'L1_S25_D1895', 'L1_S25_D1898', 'L1_S25_D1902', 'L1_S25_D1906', 'L1_S25_D1911', 'L1_S25_D1916', 'L1_S25_D1921', 'L1_S25_D1926', 'L1_S25_D1931', 'L1_S25_D1935', 'L1_S25_D1940', 'L1_S25_D1945', 'L1_S25_D1950', 'L1_S25_D1955', 'L1_S25_D1960', 'L1_S25_D1965', 'L1_S25_D1970', 'L1_S25_D1975', 'L1_S25_D1980', 'L1_S25_D1984', 'L1_S25_D1989', 'L1_S25_D1994', 'L1_S25_D1999', 'L1_S25_D2004', 'L1_S25_D2009', 'L1_S25_D2013', 'L1_S25_D2018', 'L1_S25_D2023', 'L1_S25_D2028', 'L1_S25_D2033', 'L1_S25_D2038', 'L1_S25_D2043', 'L1_S25_D2048', 'L1_S25_D2053', 'L1_S25_D2058', 'L1_S25_D2063', 'L1_S25_D2068', 'L1_S25_D2073', 'L1_S25_D2078', 'L1_S25_D2083', 'L1_S25_D2088', 'L1_S25_D2093', 'L1_S25_D2098', 'L1_S25_D2103', 'L1_S25_D2108', 'L1_S25_D2113', 'L1_S25_D2118', 'L1_S25_D2123', 'L1_S25_D2128', 'L1_S25_D2133', 'L1_S25_D2138', 'L1_S25_D2140', 'L1_S25_D2143', 'L1_S25_D2146', 'L1_S25_D2149', 'L1_S25_D2151', 'L1_S25_D2154', 'L1_S25_D2157', 'L1_S25_D2160', 'L1_S25_D2163', 'L1_S25_D2166', 'L1_S25_D2169', 'L1_S25_D2172', 'L1_S25_D2175', 'L1_S25_D2178', 'L1_S25_D2180', 'L1_S25_D2183', 'L1_S25_D2186', 'L1_S25_D2189', 'L1_S25_D2192', 'L1_S25_D2195', 'L1_S25_D2198', 'L1_S25_D2201', 'L1_S25_D2204', 'L1_S25_D2206', 'L1_S25_D2209', 'L1_S25_D2212', 'L1_S25_D2214', 'L1_S25_D2216', 'L1_S25_D2219', 'L1_S25_D2222', 'L1_S25_D2225', 'L1_S25_D2228', 'L1_S25_D2230', 'L1_S25_D2232', 'L1_S25_D2234', 'L1_S25_D2235', 'L1_S25_D2236', 'L1_S25_D2238', 'L1_S25_D2240', 'L1_S25_D2242', 'L1_S25_D2244', 'L1_S25_D2246', 'L1_S25_D2248', 'L1_S25_D2251', 'L1_S25_D2255', 'L1_S25_D2260', 'L1_S25_D2265', 'L1_S25_D2270', 'L1_S25_D2275', 'L1_S25_D2280', 'L1_S25_D2284', 'L1_S25_D2289', 'L1_S25_D2294', 'L1_S25_D2299', 'L1_S25_D2304', 'L1_S25_D2309', 'L1_S25_D2314', 'L1_S25_D2319', 'L1_S25_D2324', 'L1_S25_D2329', 'L1_S25_D2333', 'L1_S25_D2338', 'L1_S25_D2343', 'L1_S25_D2348', 'L1_S25_D2353', 'L1_S25_D2358', 'L1_S25_D2362', 'L1_S25_D2367', 'L1_S25_D2372', 'L1_S25_D2377', 'L1_S25_D2382', 'L1_S25_D2387', 'L1_S25_D2392', 'L1_S25_D2397', 'L1_S25_D2402', 'L1_S25_D2406', 'L1_S25_D2409', 'L1_S25_D2412', 'L1_S25_D2415', 'L1_S25_D2418', 'L1_S25_D2421', 'L1_S25_D2424', 'L1_S25_D2427', 'L1_S25_D2430', 'L1_S25_D2432', 'L1_S25_D2434', 'L1_S25_D2436', 'L1_S25_D2438', 'L1_S25_D2440', 'L1_S25_D2442', 'L1_S25_D2444', 'L1_S25_D2445', 'L1_S25_D2446', 'L1_S25_D2448', 'L1_S25_D2450', 'L1_S25_D2452', 'L1_S25_D2453', 'L1_S25_D2455', 'L1_S25_D2457', 'L1_S25_D2459', 'L1_S25_D2461', 'L1_S25_D2463', 'L1_S25_D2465', 'L1_S25_D2467', 'L1_S25_D2469', 'L1_S25_D2471', 'L1_S25_D2474', 'L1_S25_D2477', 'L1_S25_D2480', 'L1_S25_D2483', 'L1_S25_D2486', 'L1_S25_D2489', 'L1_S25_D2492', 'L1_S25_D2495', 'L1_S25_D2497', 'L1_S25_D2499', 'L1_S25_D2501', 'L1_S25_D2502', 'L1_S25_D2503', 'L1_S25_D2505', 'L1_S25_D2507', 'L1_S25_D2509', 'L1_S25_D2511', 'L1_S25_D2513', 'L1_S25_D2515', 'L1_S25_D2518', 'L1_S25_D2522', 'L1_S25_D2527', 'L1_S25_D2532', 'L1_S25_D2537', 'L1_S25_D2542', 'L1_S25_D2547', 'L1_S25_D2551', 'L1_S25_D2556', 'L1_S25_D2561', 'L1_S25_D2566', 'L1_S25_D2571', 'L1_S25_D2576', 'L1_S25_D2581', 'L1_S25_D2586', 'L1_S25_D2591', 'L1_S25_D2596', 'L1_S25_D2600', 'L1_S25_D2605', 'L1_S25_D2610', 'L1_S25_D2615', 'L1_S25_D2620', 'L1_S25_D2625', 'L1_S25_D2629', 'L1_S25_D2634', 'L1_S25_D2639', 'L1_S25_D2644', 'L1_S25_D2649', 'L1_S25_D2654', 'L1_S25_D2659', 'L1_S25_D2664', 'L1_S25_D2669', 'L1_S25_D2674', 'L1_S25_D2679', 'L1_S25_D2684', 'L1_S25_D2689', 'L1_S25_D2694', 'L1_S25_D2699', 'L1_S25_D2704', 'L1_S25_D2709', 'L1_S25_D2713', 'L1_S25_D2715', 'L1_S25_D2717', 'L1_S25_D2719', 'L1_S25_D2721', 'L1_S25_D2723', 'L1_S25_D2725', 'L1_S25_D2727', 'L1_S25_D2728', 'L1_S25_D2729', 'L1_S25_D2731', 'L1_S25_D2733', 'L1_S25_D2735', 'L1_S25_D2736', 'L1_S25_D2738', 'L1_S25_D2740', 'L1_S25_D2742', 'L1_S25_D2744', 'L1_S25_D2746', 'L1_S25_D2748', 'L1_S25_D2750', 'L1_S25_D2752', 'L1_S25_D2754', 'L1_S25_D2757', 'L1_S25_D2760', 'L1_S25_D2763', 'L1_S25_D2766', 'L1_S25_D2769', 'L1_S25_D2772', 'L1_S25_D2775', 'L1_S25_D2778', 'L1_S25_D2780', 'L1_S25_D2782', 'L1_S25_D2784', 'L1_S25_D2785', 'L1_S25_D2786', 'L1_S25_D2788', 'L1_S25_D2790', 'L1_S25_D2792', 'L1_S25_D2794', 'L1_S25_D2796', 'L1_S25_D2798', 'L1_S25_D2801', 'L1_S25_D2805', 'L1_S25_D2810', 'L1_S25_D2815', 'L1_S25_D2820', 'L1_S25_D2825', 'L1_S25_D2830', 'L1_S25_D2834', 'L1_S25_D2839', 'L1_S25_D2844', 'L1_S25_D2849', 'L1_S25_D2854', 'L1_S25_D2859', 'L1_S25_D2864', 'L1_S25_D2869', 'L1_S25_D2874', 'L1_S25_D2879', 'L1_S25_D2883', 'L1_S25_D2888', 'L1_S25_D2893', 'L1_S25_D2898', 'L1_S25_D2903', 'L1_S25_D2908', 'L1_S25_D2912', 'L1_S25_D2917', 'L1_S25_D2922', 'L1_S25_D2927', 'L1_S25_D2932', 'L1_S25_D2937', 'L1_S25_D2942', 'L1_S25_D2947', 'L1_S25_D2952', 'L1_S25_D2957', 'L1_S25_D2962', 'L1_S25_D2967', 'L1_S25_D2972', 'L1_S25_D2977', 'L1_S25_D2982', 'L1_S25_D2987', 'L1_S25_D2992', 'L1_S25_D2996', 'L1_S25_D2998', 'L1_S25_D3000', 'L1_S25_D3002', 'L1_S25_D3004', 'L1_S25_D3006', 'L1_S25_D3008', 'L1_S25_D3010', 'L1_S25_D3011', 'L1_S25_D3012', 'L1_S25_D3014', 'L1_S25_D3016', 'L1_S25_D3018', 'L1_S25_D3019', 'L1_S25_D3021', 'L1_S25_D3023', 'L1_S25_D3025', 'L1_S25_D3027', 'L1_S25_D3029', 'L1_S25_D3031', 'L1_S25_D3033', 'L1_S25_D3035', 'L2_S26_D3037', 'L2_S26_D3041', 'L2_S26_D3044', 'L2_S26_D3048', 'L2_S26_D3052', 'L2_S26_D3056', 'L2_S26_D3059', 'L2_S26_D3063', 'L2_S26_D3066', 'L2_S26_D3070', 'L2_S26_D3074', 'L2_S26_D3078', 'L2_S26_D3081', 'L2_S26_D3084', 'L2_S26_D3087', 'L2_S26_D3090', 'L2_S26_D3093', 'L2_S26_D3096', 'L2_S26_D3100', 'L2_S26_D3103', 'L2_S26_D3107', 'L2_S26_D3110', 'L2_S26_D3114', 'L2_S26_D3118', 'L2_S26_D3122', 'L2_S26_D3126', 'L2_S27_D3130', 'L2_S27_D3134', 'L2_S27_D3137', 'L2_S27_D3141', 'L2_S27_D3145', 'L2_S27_D3149', 'L2_S27_D3152', 'L2_S27_D3156', 'L2_S27_D3159', 'L2_S27_D3163', 'L2_S27_D3167', 'L2_S27_D3171', 'L2_S27_D3174', 'L2_S27_D3177', 'L2_S27_D3180', 'L2_S27_D3183', 'L2_S27_D3186', 'L2_S27_D3189', 'L2_S27_D3193', 'L2_S27_D3196', 'L2_S27_D3200', 'L2_S27_D3203', 'L2_S27_D3207', 'L2_S27_D3211', 'L2_S27_D3215', 'L2_S27_D3219', 'L2_S28_D3223', 'L2_S28_D3227', 'L2_S28_D3230', 'L2_S28_D3234', 'L2_S28_D3238', 'L2_S28_D3242', 'L2_S28_D3245', 'L2_S28_D3249', 'L2_S28_D3252', 'L2_S28_D3256', 'L2_S28_D3260', 'L2_S28_D3264', 'L2_S28_D3267', 'L2_S28_D3270', 'L2_S28_D3273', 'L2_S28_D3276', 'L2_S28_D3279', 'L2_S28_D3282', 'L2_S28_D3286', 'L2_S28_D3289', 'L2_S28_D3293', 'L2_S28_D3296', 'L2_S28_D3300', 'L2_S28_D3304', 'L2_S28_D3308', 'L2_S28_D3312', 'L3_S29_D3316', 'L3_S29_D3319', 'L3_S29_D3322', 'L3_S29_D3325', 'L3_S29_D3328', 'L3_S29_D3331', 'L3_S29_D3334', 'L3_S29_D3337', 'L3_S29_D3340', 'L3_S29_D3343', 'L3_S29_D3346', 'L3_S29_D3349', 'L3_S29_D3352', 'L3_S29_D3355', 'L3_S29_D3358', 'L3_S29_D3361', 'L3_S29_D3363', 'L3_S29_D3365', 'L3_S29_D3368', 'L3_S29_D3371', 'L3_S29_D3374', 'L3_S29_D3377', 'L3_S29_D3380', 'L3_S29_D3383', 'L3_S29_D3386', 'L3_S29_D3389', 'L3_S29_D3391', 'L3_S29_D3393', 'L3_S29_D3396', 'L3_S29_D3399', 'L3_S29_D3402', 'L3_S29_D3405', 'L3_S29_D3408', 'L3_S29_D3410', 'L3_S29_D3413', 'L3_S29_D3415', 'L3_S29_D3417', 'L3_S29_D3419', 'L3_S29_D3422', 'L3_S29_D3425', 'L3_S29_D3428', 'L3_S29_D3431', 'L3_S29_D3434', 'L3_S29_D3437', 'L3_S29_D3440', 'L3_S29_D3443', 'L3_S29_D3445', 'L3_S29_D3447', 'L3_S29_D3450', 'L3_S29_D3453', 'L3_S29_D3456', 'L3_S29_D3459', 'L3_S29_D3462', 'L3_S29_D3465', 'L3_S29_D3468', 'L3_S29_D3471', 'L3_S29_D3474', 'L3_S29_D3477', 'L3_S29_D3480', 'L3_S29_D3483', 'L3_S29_D3486', 'L3_S29_D3489', 'L3_S29_D3492', 'L3_S30_D3496', 'L3_S30_D3501', 'L3_S30_D3506', 'L3_S30_D3511', 'L3_S30_D3516', 'L3_S30_D3521', 'L3_S30_D3526', 'L3_S30_D3531', 'L3_S30_D3536', 'L3_S30_D3541', 'L3_S30_D3546', 'L3_S30_D3551', 'L3_S30_D3556', 'L3_S30_D3561', 'L3_S30_D3566', 'L3_S30_D3571', 'L3_S30_D3576', 'L3_S30_D3581', 'L3_S30_D3586', 'L3_S30_D3591', 'L3_S30_D3596', 'L3_S30_D3601', 'L3_S30_D3606', 'L3_S30_D3611', 'L3_S30_D3616', 'L3_S30_D3621', 'L3_S30_D3626', 'L3_S30_D3631', 'L3_S30_D3636', 'L3_S30_D3641', 'L3_S30_D3646', 'L3_S30_D3651', 'L3_S30_D3656', 'L3_S30_D3661', 'L3_S30_D3666', 'L3_S30_D3671', 'L3_S30_D3676', 'L3_S30_D3681', 'L3_S30_D3686', 'L3_S30_D3691', 'L3_S30_D3696', 'L3_S30_D3701', 'L3_S30_D3706', 'L3_S30_D3711', 'L3_S30_D3716', 'L3_S30_D3721', 'L3_S30_D3726', 'L3_S30_D3731', 'L3_S30_D3736', 'L3_S30_D3741', 'L3_S30_D3746', 'L3_S30_D3751', 'L3_S30_D3756', 'L3_S30_D3761', 'L3_S30_D3766', 'L3_S30_D3771', 'L3_S30_D3776', 'L3_S30_D3781', 'L3_S30_D3786', 'L3_S30_D3791', 'L3_S30_D3796', 'L3_S30_D3801', 'L3_S30_D3806', 'L3_S30_D3811', 'L3_S30_D3816', 'L3_S30_D3821', 'L3_S30_D3826', 'L3_S30_D3831', 'L3_S31_D3836', 'L3_S31_D3840', 'L3_S31_D3844', 'L3_S31_D3848', 'L3_S32_D3852', 'L3_S33_D3856', 'L3_S33_D3858', 'L3_S33_D3860', 'L3_S33_D3862', 'L3_S33_D3864', 'L3_S33_D3866', 'L3_S33_D3868', 'L3_S33_D3870', 'L3_S33_D3872', 'L3_S33_D3874', 'L3_S34_D3875', 'L3_S34_D3877', 'L3_S34_D3879', 'L3_S34_D3881', 'L3_S34_D3883', 'L3_S35_D3886', 'L3_S35_D3891', 'L3_S35_D3895', 'L3_S35_D3897', 'L3_S35_D3900', 'L3_S35_D3905', 'L3_S35_D3910', 'L3_S35_D3915', 'L3_S36_D3919', 'L3_S36_D3921', 'L3_S36_D3923', 'L3_S36_D3925', 'L3_S36_D3928', 'L3_S36_D3932', 'L3_S36_D3936', 'L3_S36_D3940', 'L3_S37_D3942', 'L3_S37_D3943', 'L3_S37_D3945', 'L3_S37_D3947', 'L3_S37_D3949', 'L3_S37_D3951', 'L3_S38_D3953', 'L3_S38_D3957', 'L3_S38_D3961', 'L3_S39_D3966', 'L3_S39_D3970', 'L3_S39_D3974', 'L3_S39_D3978', 'L3_S40_D3981', 'L3_S40_D3983', 'L3_S40_D3985', 'L3_S40_D3987', 'L3_S40_D3989', 'L3_S40_D3991', 'L3_S40_D3993', 'L3_S40_D3995', 'L3_S41_D3997', 'L3_S41_D3999', 'L3_S41_D4001', 'L3_S41_D4003', 'L3_S41_D4005', 'L3_S41_D4007', 'L3_S41_D4009', 'L3_S41_D4010', 'L3_S41_D4012', 'L3_S41_D4013', 'L3_S41_D4015', 'L3_S41_D4017', 'L3_S41_D4019', 'L3_S41_D4021', 'L3_S41_D4022', 'L3_S41_D4024', 'L3_S41_D4025', 'L3_S41_D4027', 'L3_S42_D4029', 'L3_S42_D4033', 'L3_S42_D4037', 'L3_S42_D4041', 'L3_S42_D4045', 'L3_S42_D4049', 'L3_S42_D4053', 'L3_S42_D4057', 'L3_S43_D4062', 'L3_S43_D4067', 'L3_S43_D4072', 'L3_S43_D4077', 'L3_S43_D4082', 'L3_S43_D4087', 'L3_S43_D4092', 'L3_S43_D4097', 'L3_S44_D4101', 'L3_S44_D4104', 'L3_S44_D4107', 'L3_S44_D4110', 'L3_S44_D4113', 'L3_S44_D4116', 'L3_S44_D4119', 'L3_S44_D4122', 'L3_S45_D4125', 'L3_S45_D4127', 'L3_S45_D4129', 'L3_S45_D4131', 'L3_S45_D4133', 'L3_S46_D4135', 'L3_S47_D4140', 'L3_S47_D4145', 'L3_S47_D4150', 'L3_S47_D4155', 'L3_S47_D4160', 'L3_S47_D4165', 'L3_S47_D4170', 'L3_S47_D4175', 'L3_S47_D4180', 'L3_S47_D4185', 'L3_S47_D4190', 'L3_S48_D4194', 'L3_S48_D4195', 'L3_S48_D4197', 'L3_S48_D4199', 'L3_S48_D4201', 'L3_S48_D4203', 'L3_S48_D4205', 'L3_S49_D4208', 'L3_S49_D4213', 'L3_S49_D4218', 'L3_S49_D4223', 'L3_S49_D4228', 'L3_S49_D4233', 'L3_S49_D4238', 'L3_S50_D4242', 'L3_S50_D4244', 'L3_S50_D4246', 'L3_S50_D4248', 'L3_S50_D4250', 'L3_S50_D4252', 'L3_S50_D4254', 'L3_S51_D4255', 'L3_S51_D4257', 'L3_S51_D4259', 'L3_S51_D4261', 'L3_S51_D4263']
            
            #Request an input file:-
            print("Getting CSV file....")
            file=request.files.get("file")


            print("Reading csvfile for num features...")
            print(type(file))
            df=pd.read_csv(file)
            df_num=pre_process_num.pre_process_num(df)
            print(type(df_num))

            print("Reading csv file for date features...")
            print(type(file))
            # df=pd.read_csv(file)
            df_date=pre_process_date.pre_process_date(df,file)
            print(type(df_date))


            #Merge_datasets:
            print("Merging processed dataset...")
            final_test=pd.merge(df_date,df_num,on="Id",how="inner")



            #Select only those features which were used during model training:--
            train_features=['L0_S0_F0', 'L0_S0_F2', 'L0_S0_F18', 'L0_S0_F20', 'L0_S1_F28', 'L0_S9_F160', 'L0_S11_F290', 'L1_S24_F1695', 'L1_S24_F1723', 'L1_S24_F1846', 'L3_S29_F3315', 'L3_S29_F3318', 'L3_S29_F3324', 'L3_S29_F3327', 'L3_S29_F3330', 'L3_S29_F3333', 'L3_S29_F3336', 'L3_S29_F3342', 'L3_S29_F3345', 'L3_S29_F3348', 'L3_S29_F3351', 'L3_S29_F3354', 'L3_S29_F3367', 'L3_S29_F3370', 'L3_S29_F3373', 'L3_S29_F3376', 'L3_S29_F3379', 'L3_S29_F3382', 'L3_S29_F3388', 'L3_S29_F3395', 'L3_S29_F3401', 'L3_S29_F3404', 'L3_S29_F3424', 'L3_S29_F3427', 'L3_S29_F3430', 'L3_S29_F3436', 'L3_S29_F3439', 'L3_S29_F3442', 'L3_S29_F3449', 'L3_S29_F3452', 'L3_S29_F3455', 'L3_S29_F3458', 'L3_S29_F3479', 'L3_S30_F3494', 'L3_S30_F3499', 'L3_S30_F3514', 'L3_S30_F3519', 'L3_S30_F3534', 'L3_S30_F3539', 'L3_S30_F3544', 'L3_S30_F3554', 'L3_S30_F3559', 'L3_S30_F3569', 'L3_S30_F3574', 'L3_S30_F3579', 'L3_S30_F3589', 'L3_S30_F3604', 'L3_S30_F3609', 'L3_S30_F3629', 'L3_S30_F3639', 'L3_S30_F3649', 'L3_S30_F3669', 'L3_S30_F3689', 'L3_S30_F3709', 'L3_S30_F3744', 'L3_S30_F3749', 'L3_S30_F3754', 'L3_S30_F3759', 'L3_S30_F3764', 'L3_S30_F3769', 'L3_S30_F3774', 'L3_S30_F3784', 'L3_S30_F3794', 'L3_S30_F3799', 'L3_S30_F3804', 'L3_S30_F3809', 'L3_S30_F3819', 'L3_S30_F3829', 'L3_S32_F3850', 'L3_S33_F3855', 'L3_S33_F3857', 'L3_S33_F3859', 'L3_S33_F3863', 'L3_S33_F3865', 'L3_S35_F3896', 'L3_S36_F3920', 'L3_S36_F3924', 'L3_S38_F3952', 'L3_S38_F3956', 'L3_S38_F3960', 'L0_S0', 'L0_S1', 'L0_S2', 'L0_S3', 'L0_S4', 'L0_S5', 'L0_S6', 'L0_S7', 'L0_S8', 'L0_S9', 'L0_S10', 'L0_S11', 'L0_S12', 'L0_S13', 'L0_S14', 'L0_S15', 'L0_S16', 'L0_S17', 'L0_S18', 'L0_S19', 'L0_S20', 'L0_S21', 'L0_S22', 'L0_S23', 'L1_S24', 'L1_S25', 'L2_S26', 'L2_S27', 'L2_S28', 'L3_S29', 'L3_S30', 'L3_S31', 'L3_S32', 'L3_S33', 'L3_S34', 'L3_S35', 'L3_S36', 'L3_S37', 'L3_S38', 'L3_S39', 'L3_S40', 'L3_S41', 'L3_S42', 'L3_S43', 'L3_S44', 'L3_S45', 'L3_S46', 'L3_S47', 'L3_S48', 'L3_S49', 'L3_S50', 'L3_S51', 'Total_time_taken', 'station_32', 'mindate', 'maxdate', 'min_time_station', 'max_time_station', 'min_Id_rev', 'min_Id', 'start_station', 'end_station']

            x_test=final_test.drop(["Id"],axis=1)
            x_test=x_test.fillna(0)
            x_test=x_test[list(train_features)]

            #Predict on merged dataset:--
            print("Loading the model...")
            filename = 'xg_boost_date_num1.pickle'

            loaded_model = pickle.load(open(filename, 'rb'))
            print("Predicting....")
            df_test_pred = loaded_model.predict(x_test)    

            #Predicted dataframe:--
            print("List of predictions....")
            result = list(df_test_pred)

            print("Dataframe of predictions....")
            result = pd.DataFrame(list(zip(final_test.Id, result)), columns=['Part ID', 'Prediction'])
            result['Prediction'] = result['Prediction'].map({1:'Defect likely', 0:'Defect Unlikely'})

            print("Sending .csv file....")

            resp = make_response(result.to_csv())
            resp.headers["Content-Disposition"] = "attachment; filename=result.csv"
            resp.headers["Content-Type"] = "text/csv"
            return resp

        else:
        	print("No match found!!")

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)



port = int(os.getenv("PORT", 8000))
if __name__ == "__main__":
    app.run(debug=True)

