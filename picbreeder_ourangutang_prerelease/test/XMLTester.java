/*
 * Unlicensed intellectual property of the University of Central Florida for
 * internal usage only. You may not distribute this code to anyone. You may
 * not use this code (as source or compiled) or information obtained from
 * this code without permission.
 *
 * Picbreeder Project
 * Evolutionary Complexity Research Group
 * School of Electrical Engineering and Computer Science
 * 2006-2007
 */

package test;

import javax.xml.parsers.*;
import java.io.*;
import org.w3c.dom.*;
import javax.xml.transform.*;
import javax.xml.transform.dom.*;
import javax.xml.transform.stream.*;
import client.evolution.*;
import client.evolution.impl.DefaultGeneticFactory;

public class XMLTester {
	public static void main(String []args) throws Exception {
		if(args.length == 0) {
			System.out.println("You must specify files to echo.");
			return;
		}
		
		for(String x : args)
			test(x);
	}
	
	public static void test(String file) throws Exception {
		DocumentBuilder builder = DocumentBuilderFactory.newInstance().newDocumentBuilder();
		Document doc = builder.parse(new File(file));
		
		Genome g = GeneticFactoryInstance.get().createInvalidGenome();
		g.load(doc.getDocumentElement());
		
		doc = builder.newDocument();
		Element root = doc.createElement(g.getElementName());
		doc.appendChild(root);
		g.store(root, doc);
		
		TransformerFactory transFactory = TransformerFactory.newInstance();
		transFactory.setAttribute("indent-number", 4);
		Transformer trans = transFactory.newTransformer();
		trans.setOutputProperty(OutputKeys.INDENT, "yes");
		trans.transform(new DOMSource(doc), new StreamResult(new OutputStreamWriter(System.out)));
		trans.transform(new DOMSource(doc), new StreamResult(new OutputStreamWriter(new FileOutputStream(new File("test.xml")))));
		
		System.out.println();
		System.out.println();
	}
}
